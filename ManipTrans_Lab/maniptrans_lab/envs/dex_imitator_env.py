"""
Isaac Lab port of
    ManipTrans/maniptrans_envs/lib/envs/tasks/dexhandimitator.py
    (DexHandImitatorRHEnv / DexHandImitatorLHEnv)

Structure mirrors MimicKit's DeepMimic-style envs (one file per task type)
but the content — demo data loading, proprio/privileged/target obs split,
imitation reward, noisy demo-state reset — is a 1-to-1 port of ManipTrans's
imitator task.

Sections marked with `# PORT:` are direct translations of an Isaac-Gym code
block to the engine-API equivalent. Sections marked with `# TODO(port):`
are deep task internals (e.g. the full JIT-scripted imitation reward) where
the intent is to drop in the ManipTrans implementation unchanged once the
dependency on isaacgym.torch_utils.{quat_mul, quat_conjugate, normalize_angle}
is satisfied by utils/torch_jit_utils.py (already copied).
"""
from __future__ import annotations

import os
import numpy as np
import torch
from tqdm import tqdm

import gymnasium.spaces as spaces

import engines.engine as engine
from util.logger import Logger

import envs.dex_env as dex_env
import envs.base_env as base_env
from envs.base_env import DoneFlags

from dataset.factory import ManipDataFactory
from dataset.transform import (
    aa_to_quat, aa_to_rotmat, quat_to_rotmat, rotmat_to_quat,
)
from utils import torch_jit_utils


class DexImitatorEnv(dex_env.DexEnv):
    """
    Single-hand imitation-learning task.

    Observations (dict):
        proprioception: q, cos(q), sin(q), base_state (wrist pos/rot/vel/ang_vel, pos masked)
        privileged:    dq (joint velocities)
        target:        future wrist pose / vel + future joint pos/vel from the demo
    """

    NAME = "dex_imitator"

    def __init__(self, env_config, engine_config, num_envs, device,
                 visualize, record_video=False):

        # --- task-specific params ---
        self._max_episode_length = env_config["episodeLength"]
        self._data_indices = env_config["dataIndices"]
        self._obs_future_length = env_config.get("obsFutureLength", 3)
        self._rollout_state_init = env_config.get("rolloutStateInit", False)
        self._random_state_init = env_config.get("randomStateInit", True)
        # Reset noise knobs (set to 0.0 to reproduce retargeted pose exactly).
        self._reset_wrist_pos_noise_std = float(env_config.get("resetWristPosNoiseStd", 0.01))
        self._reset_wrist_rot_noise_rad = float(env_config.get("resetWristRotNoiseRad", (np.pi / 18)))
        self._reset_wrist_vel_noise_std = float(env_config.get("resetWristVelNoiseStd", 0.01))
        # Optional reset diagnostics for debugging palm orientation/position mismatch.
        self._debug_reset_error = bool(env_config.get("debugResetError", False))
        self._debug_reset_error_interval = int(env_config.get("debugResetErrorInterval", 200))
        self._debug_reset_counter = 0
        # Test-time termination diagnostics (FAIL/SUCC/TIME).
        self._print_done_reason_in_test = bool(env_config.get("printDoneReasonInTest", True))
        # Test-time overlay: draw demo(retarget) skeleton with current hand.
        self._draw_retarget_skeleton_in_test = bool(env_config.get("drawRetargetSkeletonInTest", True))
        self._draw_retarget_skeleton_max_envs = int(env_config.get("drawRetargetSkeletonMaxEnvs", 4))
        self._draw_retarget_skeleton_line_width = float(env_config.get("drawRetargetSkeletonLineWidth", 2.0))
        self._draw_retarget_skeleton_warned = False
        # Failure gate/tolerance knobs to avoid premature episode truncation.
        self._fail_grace_steps_train = int(env_config.get("failGraceStepsTrain", 20))
        self._fail_grace_steps_test = int(env_config.get("failGraceStepsTest", self._fail_grace_steps_train))
        self._fail_distance_scale = float(env_config.get("failDistanceScale", 1.0))
        # In test/visualization, deterministic reset is usually preferred so
        # frame-0 pose matches retarget viewer exactly.
        self._deterministic_reset = False
        self._det_reset_in_test = env_config.get("deterministicResetInTest", True)
        self._random_state_init_cfg = self._random_state_init
        self._random_state_init_test = env_config.get("randomStateInitInTest", False)
        self._prop_obs_keys = env_config.get("obsKeys",
                                             ["q", "cos_q", "sin_q", "base_state"])
        self._priv_obs_keys = env_config.get("privilegedObsKeys", ["dq"])
        self._env_cfg = env_config  # kept so _update_reward can read tightenFactor
        self._table_asset = env_config.get("table_file", "")
        self._table_size = np.array(env_config.get("tableSize", [0.8, 0.8, 0.05]), dtype=np.float32)
        self._table_surface_z = float(env_config.get("tableSurfaceZ", dex_env.TABLE_SURFACE_Z))
        # Match legacy table asset spawn center used in _build_scene_props.
        self._table_center_xy = np.array(env_config.get("tableCenterXY", [-0.1, 0.0]), dtype=np.float32)
        self._table_from_asset = bool(self._table_asset and os.path.exists(self._table_asset))

        # mujoco → gym frame transform (table origin, z-up)
        m2g = np.eye(4)
        m2g[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
        m2g[:3, 3] = np.array([0.0, 0.0, dex_env.TABLE_SURFACE_Z])
        self._mujoco2gym_transf_np = m2g

        super().__init__(env_config=env_config,
                         engine_config=engine_config,
                         num_envs=num_envs,
                         device=device,
                         visualize=visualize,
                         record_video=record_video)

        if not self._table_from_asset:
            Logger.print(
                "[DexImitatorEnv] table_file missing; building procedural kinematic tables per env."
            )
            self._build_procedural_tables()

        # demo data now that self._device and self._dexhand are ready
        self._load_demo_data(env_config)

        # reset once to populate buffers
        self._reset_envs(self._env_ids)
        return

    def set_mode(self, mode):
        super().set_mode(mode)
        if mode == base_env.EnvMode.TEST:
            self._deterministic_reset = bool(self._det_reset_in_test)
            self._random_state_init = bool(self._random_state_init_test)
        else:
            self._deterministic_reset = False
            self._random_state_init = bool(self._random_state_init_cfg)
        return

    # ------------------------------------------------------------------ #
    # Scene construction                                                  #
    # ------------------------------------------------------------------ #
    def _build_scene_props(self, env_id, env_config):
        """PORT: ManipTrans dexhandimitator._create_envs — table only (imitator has no manip object)."""
        # Use USD asset when available. When missing, we defer to procedural
        # table creation after simulator initialization.
        if self._table_from_asset:
            table_id = self._engine.create_obj(
                env_id=env_id,
                obj_type=engine.ObjType.rigid,
                asset_file=self._table_asset,
                name="table",
                is_visual=False,
                fix_root=True,
                start_pos=np.array([-0.1, 0.0, 0.4]),
                start_rot=np.array([0.0, 0.0, 0.0, 1.0]),
                color=np.array([0.1, 0.1, 0.1]),
            )
            if env_id == 0:
                self._table_obj_ids.append(table_id)
        return

    def _build_procedural_tables(self):
        for env_id in range(self.get_num_envs()):
            self._build_table(env_id)
        return

    def _build_table(self, env_id):
        """Add a kinematic box (table surface) to each env when table USD is absent."""
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

        stage = self._engine._stage
        env_offsets = self._engine._env_offsets

        l, w, h = self._table_size
        surface_z = self._table_surface_z
        center_z = surface_z - h / 2.0

        ox = float(env_offsets[env_id, 0].item()) + float(self._table_center_xy[0])
        oy = float(env_offsets[env_id, 1].item()) + float(self._table_center_xy[1])

        table_path = "/World/table_env{:d}".format(env_id)

        xform_prim = stage.DefinePrim(table_path, "Xform")
        cube_prim = stage.DefinePrim(table_path + "/Cube", "Cube")

        xformable = UsdGeom.Xformable(xform_prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(ox, oy, center_z))
        xformable.AddScaleOp().Set(Gf.Vec3d(float(l) / 2.0, float(w) / 2.0, float(h) / 2.0))

        UsdGeom.Cube(cube_prim).GetSizeAttr().Set(2.0)
        UsdPhysics.CollisionAPI.Apply(cube_prim)
        UsdPhysics.RigidBodyAPI.Apply(xform_prim)
        UsdPhysics.RigidBodyAPI.Get(stage, table_path).GetKinematicEnabledAttr().Set(True)

        physics_material_path = table_path + "/PhysicsMaterial"
        mat_prim = stage.DefinePrim(physics_material_path, "Material")
        UsdPhysics.MaterialAPI.Apply(mat_prim)
        phys_mat = UsdPhysics.MaterialAPI.Get(stage, physics_material_path)
        phys_mat.GetStaticFrictionAttr().Set(0.1)
        phys_mat.GetDynamicFrictionAttr().Set(0.1)
        phys_mat.GetRestitutionAttr().Set(0.0)

        mat = UsdShade.Material.Get(stage, physics_material_path)
        UsdShade.MaterialBindingAPI.Apply(cube_prim).Bind(
            mat, UsdShade.Tokens.strongerThanDescendants, "physics"
        )

        preview_shader = UsdShade.Shader.Define(stage, table_path + "/Cube/PreviewSurface")
        preview_shader.CreateIdAttr("UsdPreviewSurface")
        preview_shader.CreateInput(
            "diffuseColor", Sdf.ValueTypeNames.Color3f
        ).Set(Gf.Vec3f(0.65, 0.45, 0.25))
        return

    # ------------------------------------------------------------------ #
    # Demo-data loading                                                   #
    # ------------------------------------------------------------------ #
    def _load_demo_data(self, env_config):
        mujoco2gym_transf = torch.tensor(
            self._mujoco2gym_transf_np, device=self._device, dtype=torch.float32,
        )

        dataset_types = list({ManipDataFactory.dataset_type(i) for i in self._data_indices})
        self._demo_dataset_dict = {}
        for dtype in dataset_types:
            self._demo_dataset_dict[dtype] = ManipDataFactory.create_data(
                manipdata_type=dtype,
                side=self._side,
                device=self._device,
                mujoco2gym_transf=mujoco2gym_transf,
                max_seq_len=self._max_episode_length,
                dexhand=self._dexhand,
                embodiment=self._dexhand_type,
            )

        def segment_data(k):
            idx = self._data_indices[k % len(self._data_indices)]
            return self._demo_dataset_dict[ManipDataFactory.dataset_type(idx)][idx]

        raw = [segment_data(i) for i in tqdm(range(self.get_num_envs()),
                                             desc="loading demo sequences")]
        self._demo_data = self._pack_data(raw)
        return

    def _pack_data(self, data):
        """PORT: dexhandimitator.pack_data — pad sequences to max_len, stack."""
        num_envs = self.get_num_envs()
        packed = {}
        packed["seq_len"] = torch.tensor([len(d["obj_trajectory"]) for d in data],
                                         device=self._device)
        max_len = packed["seq_len"].max()
        assert max_len <= self._max_episode_length

        def fill(stack):
            for i in range(len(stack)):
                if len(stack[i]) < max_len:
                    pad = stack[i][-1].unsqueeze(0).repeat(
                        max_len - len(stack[i]), *[1 for _ in stack[i].shape[1:]])
                    stack[i] = torch.cat([stack[i], pad], dim=0)
            # NOTE: do NOT call .squeeze() here — that would collapse the batch
            # dimension when num_envs == 1 and break downstream advanced indexing
            # like demo_data[key][env_ids, seq_idx]. Keep shape [N, T, ...].
            return torch.stack(stack)

        for k in data[0].keys():
            if "alt" in k:
                continue
            if k in ("mano_joints", "mano_joints_velocity"):
                stack = []
                for d in data:
                    entry = d[k]
                    if isinstance(entry, dict):
                        cols = []
                        for jn in self._dexhand.body_names:
                            mano_names = self._dexhand.to_hand(jn)
                            if not mano_names or mano_names[0] == "wrist":
                                continue
                            if mano_names[0] not in entry:
                                # data source missing this MANO joint — zero-fill so
                                # shape stays consistent across sequences.
                                T = next(iter(entry.values())).shape[0]
                                cols.append(torch.zeros(T, 3, device=self._device))
                            else:
                                cols.append(entry[mano_names[0]])
                        stack.append(torch.concat(cols, dim=-1))
                    elif isinstance(entry, torch.Tensor):
                        # already concatenated [T, (n_bodies-1)*3] in dex-body order
                        stack.append(entry.reshape(entry.shape[0], -1))
                    else:
                        raise TypeError(f"Unsupported mano_joints type: {type(entry)}")
                packed[k] = fill(stack)
            elif isinstance(data[0][k], torch.Tensor):
                stack = [d[k] for d in data]
                # obj_verts: [N, V, 3] (kept as [N, V, 3] so indexing is consistent).
                # Everything else: time-sequence, padded to max_len -> [N, T, ...].
                packed[k] = torch.stack(stack) if k == "obj_verts" else fill(stack)
            else:
                packed[k] = [d[k] for d in data]
        return packed

    # ------------------------------------------------------------------ #
    # Observation                                                         #
    # ------------------------------------------------------------------ #
    def _obs_dims_dict(self):
        n_dof = self._dexhand.n_dofs
        # proprioception: q (n_dof) + cos_q (n_dof) + sin_q (n_dof) + base_state (13, pos masked to 0)
        prop_dim = 3 * n_dof + 13
        priv_dim = n_dof if "dq" in self._priv_obs_keys else 0
        # target: (3 delta_wrist_pos + 3 wrist_vel + 3 delta_wrist_vel + 4 wrist_quat + 4 delta_wrist_quat
        #          + 3 wrist_ang_vel + 3 delta_wrist_ang_vel + (n_bodies-1)*(3 delta_jpos + 3 jvel + 3 delta_jvel))
        #         × obs_future_length
        per_frame = 3 + 3 + 3 + 4 + 4 + 3 + 3 + (self._dexhand.n_bodies - 1) * 9
        tar_dim = per_frame * self._obs_future_length
        return {"proprioception": prop_dim, "privileged": priv_dim, "target": tar_dim}

    def _build_action_space(self):
        # (optional quat_norm) + 6 (wrist pose) + n_dof + (optional 3 for PID)
        n_dof = self._dexhand.n_dofs
        extra = (1 if self._use_quat_rot else 0) + (3 if self._use_pid_control else 0)
        action_size = extra + 6 + n_dof
        low = -np.ones(action_size, dtype=np.float32)
        high = np.ones(action_size, dtype=np.float32)
        return spaces.Box(low=low, high=high)

    def _compute_obs(self, env_ids=None):
        """PORT: dexhandimitator.compute_observations.
        Returns dict with 'proprioception', ('privileged'), 'target' keys."""
        dex_id = self._get_dexhand_obj_id()

        q = self._engine.get_dof_pos(dex_id)
        dq = self._engine.get_dof_vel(dex_id)
        root_pos = self._engine.get_root_pos(dex_id)
        root_rot = self._engine.get_root_rot(dex_id)
        root_vel = self._engine.get_root_vel(dex_id)
        root_ang_vel = self._engine.get_root_ang_vel(dex_id)

        # base_state is [pos(3), quat(4), lin_vel(3), ang_vel(3)], but ManipTrans
        # zeros the position component before feeding to the policy.
        zero_pos = torch.zeros_like(root_pos)
        base_state = torch.cat([zero_pos, root_rot, root_vel, root_ang_vel], dim=-1)

        prop = torch.cat([q, torch.cos(q), torch.sin(q), base_state], dim=-1)

        out = {"proprioception": prop}
        if "dq" in self._priv_obs_keys:
            out["privileged"] = dq

        out["target"] = self._compute_target_obs(root_pos, root_rot, root_vel, root_ang_vel)

        if env_ids is not None:
            out = {k: v[env_ids] for k, v in out.items()}
        return out

    def _compute_target_obs(self, cur_wrist_pos, cur_wrist_rot, cur_wrist_vel, cur_wrist_ang_vel):
        """PORT: dexhandimitator.compute_observations — target construction block.
        cur_wrist_rot is xyzw."""
        nE = self.get_num_envs()
        nF = self._obs_future_length

        # Future-frame indices, safely clamped so that cur_idx + t stays in
        # [0, seq_len-1] for every t in range(nF). When progress_buf is near
        # the end of the trajectory (e.g. random-state-init picked seq_idx=57,
        # seq_len=60) the stacked indices would otherwise overflow and trip
        # a device-side assert in torch.gather.
        seq_len = self._demo_data["seq_len"]
        cur_idx = torch.clamp(
            self._progress_buf + 1,
            torch.zeros_like(seq_len),
            seq_len - 1,
        )
        cur_idx = torch.stack([cur_idx + t for t in range(nF)], dim=-1)  # [B, K]
        # Re-clamp every future step so the +t offsets never overflow.
        cur_idx = torch.minimum(cur_idx, (seq_len - 1).unsqueeze(-1))

        nT = self._demo_data["wrist_pos"].shape[1]

        def indicing(data, idx):
            assert data.shape[0] == nE and data.shape[1] == nT
            remaining_shape = data.shape[2:]
            expanded = idx
            for _ in remaining_shape:
                expanded = expanded.unsqueeze(-1)
            expanded = expanded.expand(-1, -1, *remaining_shape)
            return torch.gather(data, 1, expanded)

        # wrist position / velocity targets ------------------------------------
        t_wrist_pos = indicing(self._demo_data["wrist_pos"], cur_idx)              # [B, K, 3]
        delta_wrist_pos = (t_wrist_pos - cur_wrist_pos[:, None]).reshape(nE, -1)

        t_wrist_vel = indicing(self._demo_data["wrist_velocity"], cur_idx)
        wrist_vel_flat = t_wrist_vel.reshape(nE, -1)
        delta_wrist_vel = (t_wrist_vel - cur_wrist_vel[:, None]).reshape(nE, -1)

        # wrist rotation target --------------------------------------------------
        t_wrist_rot_aa = indicing(self._demo_data["wrist_rot"], cur_idx)           # axis-angle
        wrist_quat_wxyz = aa_to_quat(t_wrist_rot_aa.reshape(nE * nF, -1))          # wxyz
        wrist_quat_xyzw = wrist_quat_wxyz[:, [1, 2, 3, 0]]
        delta_wrist_quat = torch_jit_utils.quat_mul(
            cur_wrist_rot[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            torch_jit_utils.quat_conjugate(wrist_quat_xyzw),
        ).reshape(nE, -1)
        wrist_quat_flat = wrist_quat_xyzw.reshape(nE, -1)

        t_wrist_ang_vel = indicing(self._demo_data["wrist_angular_velocity"], cur_idx)
        wrist_ang_vel_flat = t_wrist_ang_vel.reshape(nE, -1)
        delta_wrist_ang_vel = (t_wrist_ang_vel - cur_wrist_ang_vel[:, None]).reshape(nE, -1)

        # joint-level targets ----------------------------------------------------
        t_jpos = indicing(self._demo_data["mano_joints"], cur_idx).reshape(nE, nF, -1, 3)
        body_pos = self._engine.get_body_pos(self._get_dexhand_obj_id())           # [B, nBody, 3]
        cur_joint_pos = body_pos[:, 1:, :]                                         # skip wrist (body 0)
        delta_jpos = (t_jpos - cur_joint_pos[:, None]).reshape(nE, -1)

        t_jvel = indicing(self._demo_data["mano_joints_velocity"], cur_idx).reshape(nE, nF, -1, 3)
        body_vel = self._engine.get_body_vel(self._get_dexhand_obj_id())
        cur_joint_vel = body_vel[:, 1:, :]
        jvel_flat = t_jvel.reshape(nE, -1)
        delta_jvel = (t_jvel - cur_joint_vel[:, None]).reshape(nE, -1)

        target = torch.cat([
            delta_wrist_pos, wrist_vel_flat, delta_wrist_vel,
            wrist_quat_flat, delta_wrist_quat,
            wrist_ang_vel_flat, delta_wrist_ang_vel,
            delta_jpos, jvel_flat, delta_jvel,
        ], dim=-1)
        return target

    # ------------------------------------------------------------------ #
    # Reward + done (PORT: ManipTrans compute_imitation_reward)           #
    # ------------------------------------------------------------------ #
    def _build_state_and_target(self):
        """Assemble the `states` and `target_states` dicts the original
        compute_imitation_reward expects. Kept as a regular (non-JIT) helper
        because our engine getters return Python tensors, not the jit-friendly
        cached tensors of the original."""
        dex_id = self._get_dexhand_obj_id()
        nE = self.get_num_envs()
        bs = torch.arange(nE, device=self._device)

        # --- current sim states ------------------------------------------
        q = self._engine.get_dof_pos(dex_id)                 # [B, n_dof]
        dq = self._engine.get_dof_vel(dex_id)                # [B, n_dof]
        root_pos = self._engine.get_root_pos(dex_id)         # [B, 3]
        root_rot = self._engine.get_root_rot(dex_id)         # [B, 4] xyzw
        root_vel = self._engine.get_root_vel(dex_id)         # [B, 3]
        root_ang = self._engine.get_root_ang_vel(dex_id)     # [B, 3]
        base_state = torch.cat([root_pos, root_rot, root_vel, root_ang], dim=-1)   # [B, 13]

        body_pos = self._engine.get_body_pos(dex_id)         # [B, n_body, 3]
        body_rot = self._engine.get_body_rot(dex_id)         # [B, n_body, 4]
        body_vel = self._engine.get_body_vel(dex_id)         # [B, n_body, 3]
        body_ang = self._engine.get_body_ang_vel(dex_id)     # [B, n_body, 3]
        joints_state = torch.cat([body_pos, body_rot, body_vel, body_ang], dim=-1)  # [B, n_body, 13]

        states = {
            "q": q,
            "dq": dq,
            "base_state": base_state,
            "joints_state": joints_state,
        }

        # --- demo target states at current progress ---------------------
        # Clamp to [0, seq_len-1] so reward/target indexing is safe even when
        # progress_buf has just been flagged succeeded/failed but obs is still
        # being computed in the same step.
        cur_idx = torch.minimum(self._progress_buf, self._demo_data["seq_len"] - 1)
        tar_wrist_pos = self._demo_data["wrist_pos"][bs, cur_idx]
        tar_wrist_rot_aa = self._demo_data["wrist_rot"][bs, cur_idx]
        tar_wrist_quat_xyzw = aa_to_quat(tar_wrist_rot_aa)[:, [1, 2, 3, 0]]

        tar_wrist_vel = self._demo_data["wrist_velocity"][bs, cur_idx]
        tar_wrist_ang_vel = self._demo_data["wrist_angular_velocity"][bs, cur_idx]

        tar_joints_pos = self._demo_data["mano_joints"][bs, cur_idx].reshape(nE, -1, 3)
        tar_joints_vel = self._demo_data["mano_joints_velocity"][bs, cur_idx].reshape(nE, -1, 3)

        # --- power terms (computed in-env, not demo) -------------------
        dof_torque = self._engine.get_dof_forces(dex_id)
        power = torch.abs(dof_torque * dq).sum(dim=-1)
        # wrist_power mirrors ManipTrans's dexhandimitator.py:582-595:
        #   |apply_forces_wrist · linear_vel| + |apply_torque_wrist · ang_vel|
        # DexEnv._apply_wrist_force caches the currently-applied wrench in
        # _wrist_force_buf / _wrist_torque_buf (post-moving-average). Before
        # the first action is applied these don't exist yet.
        if hasattr(self, "_wrist_force_buf"):
            wrist_power = torch.abs(
                torch.sum(self._wrist_force_buf * root_vel, dim=-1)
            ) + torch.abs(
                torch.sum(self._wrist_torque_buf * root_ang, dim=-1)
            )
        else:
            wrist_power = torch.zeros(nE, device=self._device, dtype=torch.float)

        target_states = {
            "wrist_pos": tar_wrist_pos,
            "wrist_quat": tar_wrist_quat_xyzw,
            "wrist_vel": tar_wrist_vel,
            "wrist_ang_vel": tar_wrist_ang_vel,
            "joints_pos": tar_joints_pos,
            "joints_vel": tar_joints_vel,
            "power": power,
            "wrist_power": wrist_power,
        }
        return states, target_states

    def _update_reward(self):
        """Port of ManipTrans's compute_imitation_reward (JIT function at the
        bottom of maniptrans_envs/lib/envs/tasks/dexhandimitator.py). Non-JIT
        here because we don't need the speedup (per-iteration call count is
        small) and JIT would constrain us to tuple-typed weight_idx."""
        states, target_states = self._build_state_and_target()

        cur_eef_pos = states["base_state"][:, :3]
        cur_eef_quat = states["base_state"][:, 3:7]
        cur_eef_vel = states["base_state"][:, 7:10]
        cur_eef_ang = states["base_state"][:, 10:13]

        tar_eef_pos = target_states["wrist_pos"]
        tar_eef_quat = target_states["wrist_quat"]
        tar_eef_vel = target_states["wrist_vel"]
        tar_eef_ang = target_states["wrist_ang_vel"]

        diff_eef_pos_dist = torch.norm(tar_eef_pos - cur_eef_pos, dim=-1)
        diff_eef_vel = tar_eef_vel - cur_eef_vel
        diff_eef_ang = tar_eef_ang - cur_eef_ang

        joints_pos = states["joints_state"][:, 1:, :3]
        joints_vel = states["joints_state"][:, 1:, 7:10]
        tar_joints_pos = target_states["joints_pos"]
        tar_joints_vel = target_states["joints_vel"]

        diff_joints_pos_dist = torch.norm(tar_joints_pos - joints_pos, dim=-1)
        diff_joints_vel = tar_joints_vel - joints_vel

        wi = self._dexhand.weight_idx
        def sel(name):
            # weight_idx is keyed with 0=wrist included; the JIT code subtracts 1
            # to index joints_pos which excludes the wrist.
            ids = [k - 1 for k in wi[name]]
            return diff_joints_pos_dist[:, ids].mean(dim=-1)

        d_thumb  = sel("thumb_tip")
        d_index  = sel("index_tip")
        d_middle = sel("middle_tip")
        d_ring   = sel("ring_tip")
        d_pinky  = sel("pinky_tip")
        d_lvl1   = sel("level_1_joints")
        d_lvl2   = sel("level_2_joints")

        # --- reward components (verbatim weights from ManipTrans) ---------
        r_eef_pos    = torch.exp(-40 * diff_eef_pos_dist)
        r_thumb_tip  = torch.exp(-100 * d_thumb)
        r_index_tip  = torch.exp(-90  * d_index)
        r_middle_tip = torch.exp(-80  * d_middle)
        r_pinky_tip  = torch.exp(-60  * d_pinky)
        r_ring_tip   = torch.exp(-60  * d_ring)
        r_lvl1       = torch.exp(-50  * d_lvl1)
        r_lvl2       = torch.exp(-40  * d_lvl2)
        r_eef_vel    = torch.exp(-1   * diff_eef_vel.abs().mean(dim=-1))
        r_eef_ang    = torch.exp(-1   * diff_eef_ang.abs().mean(dim=-1))
        r_jnt_vel    = torch.exp(-1   * diff_joints_vel.abs().mean(dim=-1).mean(dim=-1))

        # Robustify quaternion math:
        # Isaac-state quaternions can occasionally drift from unit length, and
        # quat_to_angle_axis assumes normalized input. Without this, rare
        # out-of-range acos/sqrt can inject NaN into reward.
        cur_eef_quat = torch_jit_utils.normalize(cur_eef_quat)
        tar_eef_quat = torch_jit_utils.normalize(tar_eef_quat)
        diff_eef_rot = torch_jit_utils.quat_mul(
            tar_eef_quat, torch_jit_utils.quat_conjugate(cur_eef_quat))
        diff_eef_rot = torch_jit_utils.normalize(diff_eef_rot)
        diff_eef_rot_angle, _ = torch_jit_utils.quat_to_angle_axis(diff_eef_rot)
        r_eef_rot = torch.exp(-1 * diff_eef_rot_angle.abs())

        r_power       = torch.exp(-10 * target_states["power"])
        r_wrist_power = torch.exp(-2  * target_states["wrist_power"])

        reward = (
            0.1  * r_eef_pos
            + 0.6  * r_eef_rot
            + 0.9  * r_thumb_tip
            + 0.8  * r_index_tip
            + 0.75 * r_middle_tip
            + 0.6  * r_pinky_tip
            + 0.6  * r_ring_tip
            + 0.5  * r_lvl1
            + 0.3  * r_lvl2
            + 0.1  * r_eef_vel
            + 0.05 * r_eef_ang
            + 0.1  * r_jnt_vel
            + 0.5  * r_power
            + 0.5  * r_wrist_power
        )

        # --- failure / success signals (kept as attrs for _update_done) ---
        cur_dof_vel = states["dq"]
        finite_ok = (
            torch.isfinite(cur_eef_pos).all(dim=-1)
            & torch.isfinite(cur_eef_quat).all(dim=-1)
            & torch.isfinite(cur_eef_vel).all(dim=-1)
            & torch.isfinite(cur_eef_ang).all(dim=-1)
            & torch.isfinite(joints_pos).all(dim=(-2, -1))
            & torch.isfinite(joints_vel).all(dim=(-2, -1))
            & torch.isfinite(cur_dof_vel).all(dim=-1)
            & torch.isfinite(reward)
        )
        invalid_state = ~finite_ok
        error_buf = (
            (torch.norm(cur_eef_vel, dim=-1) > 100)
            | (torch.norm(cur_eef_ang, dim=-1) > 200)
            | (torch.norm(joints_vel, dim=-1).mean(dim=-1) > 100)
            | (cur_dof_vel.abs().mean(dim=-1) > 200)
        ) | invalid_state

        # scale_factor: imitation-difficulty schedule from YAML.
        # Matches ManipTrans dexhandimitator.compute_reward curriculum.
        scale = self._compute_tighten_scale_factor()

        fail_grace_steps = (
            self._fail_grace_steps_test
            if self._mode == base_env.EnvMode.TEST
            else self._fail_grace_steps_train
        )
        fail_dist_scale = max(self._fail_distance_scale, 1.0e-6)

        failed = (
            (
                (d_thumb  > (0.04 / 0.7 * scale * fail_dist_scale))
                | (d_index  > (0.045 / 0.7 * scale * fail_dist_scale))
                | (d_middle > (0.05 / 0.7 * scale * fail_dist_scale))
                | (d_pinky  > (0.06 / 0.7 * scale * fail_dist_scale))
                | (d_ring   > (0.06 / 0.7 * scale * fail_dist_scale))
                | (d_lvl1   > (0.07 / 0.7 * scale * fail_dist_scale))
                | (d_lvl2   > (0.08 / 0.7 * scale * fail_dist_scale))
            )
            & (self._running_progress_buf >= fail_grace_steps)
        ) | error_buf

        succeeded = (
            (self._progress_buf + 1 + 3 >= self._demo_data["seq_len"])
            & ~failed
        )

        # Never allow NaN/Inf to enter rollout buffers (breaks TB + PPO stats).
        reward = torch.where(
            invalid_state,
            torch.zeros_like(reward),
            torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0),
        )

        self._reward_buf[:] = reward
        self._failure_buf[:] = failed.long()
        self._success_buf[:] = succeeded.long()
        return

    def _compute_tighten_scale_factor(self):
        if not self._training:
            return 1.0

        method = self._env_cfg.get("tightenMethod", "None")
        factor = float(self._env_cfg.get("tightenFactor", 1.0))
        steps = max(int(self._env_cfg.get("tightenSteps", 1)), 1)
        last_step = int(self._timestep_buf.max().item())

        if method == "None":
            return 1.0
        if method == "const":
            return factor
        if method == "linear_decay":
            return 1 - (1 - factor) / steps * min(last_step, steps)
        if method == "exp_decay":
            return (np.e * 2) ** (-1 * last_step / steps) * (1 - factor) + factor
        if method == "cos":
            return factor + np.abs(
                -1 * (1 - factor) * np.cos(last_step / steps * np.pi)
            ) * (2 ** (-1 * last_step / steps))
        raise NotImplementedError(f"Unsupported tightenMethod: {method}")

    def _update_done(self):
        """Dones computed from the failure/success flags set by _update_reward."""
        done = torch.full_like(self._done_buf, DoneFlags.NULL.value)
        done[self._failure_buf.bool()] = DoneFlags.FAIL.value
        done[self._success_buf.bool()] = DoneFlags.SUCC.value
        # time-out as a safety net
        time_out = self._progress_buf >= (self._demo_data["seq_len"] - 1)
        done = torch.where((done == DoneFlags.NULL.value) & time_out,
                           torch.full_like(done, DoneFlags.TIME.value), done)
        self._done_buf[:] = done
        self._log_done_reason(done)
        return

    def _log_done_reason(self, done):
        if self._mode != base_env.EnvMode.TEST or (not self._print_done_reason_in_test):
            return

        done_ids = torch.nonzero(done != DoneFlags.NULL.value, as_tuple=False).flatten()
        if done_ids.numel() == 0:
            return

        reason_map = {
            DoneFlags.FAIL.value: "FAIL",
            DoneFlags.SUCC.value: "SUCC",
            DoneFlags.TIME.value: "TIME",
        }
        done_cpu = done[done_ids].detach().cpu().tolist()
        env_ids_cpu = done_ids.detach().cpu().tolist()
        running_prog_cpu = self._running_progress_buf[done_ids].detach().cpu().tolist()
        progress_cpu = self._progress_buf[done_ids].detach().cpu().tolist()

        for env_id, done_flag, running_prog, progress in zip(
            env_ids_cpu, done_cpu, running_prog_cpu, progress_cpu
        ):
            reason = reason_map.get(int(done_flag), f"UNKNOWN({int(done_flag)})")
            Logger.print(
                f"[DoneReason] env={env_id} reason={reason} "
                f"running_progress={int(running_prog)} progress={int(progress)}"
            )

    # ------------------------------------------------------------------ #
    # Reset                                                               #
    # ------------------------------------------------------------------ #
    def _render_scene(self):
        # Engine render clears old debug lines each frame. Draw overlay after the
        # render call so it remains visible for the next frame.
        super()._render_scene()
        if not self._should_draw_retarget_skeleton():
            return

        try:
            self._draw_retarget_skeleton_overlay()
        except Exception as e:
            if not self._draw_retarget_skeleton_warned:
                Logger.print(f"[RetargetOverlay] disabled due to error: {e}")
                self._draw_retarget_skeleton_warned = True
        return

    def _should_draw_retarget_skeleton(self):
        if self._mode != base_env.EnvMode.TEST:
            return False
        if not self._draw_retarget_skeleton_in_test:
            return False
        if not hasattr(self, "_demo_data"):
            return False
        if not hasattr(self._dexhand, "bone_links") or self._dexhand.bone_links is None:
            return False
        return len(self._dexhand.bone_links) > 0

    def _draw_retarget_skeleton_overlay(self):
        dex_id = self._get_dexhand_obj_id()
        max_envs = min(self.get_num_envs(), max(0, self._draw_retarget_skeleton_max_envs))
        if max_envs <= 0:
            return

        # Bone links are in common body order where index 0 is wrist.
        links = np.asarray(self._dexhand.bone_links, dtype=np.int64)
        if links.ndim != 2 or links.shape[1] != 2:
            return

        env_ids = torch.arange(max_envs, device=self._device, dtype=torch.long)
        seq_idx = torch.minimum(self._progress_buf[env_ids], self._demo_data["seq_len"][env_ids] - 1)

        # Current sim skeleton [B, n_body, 3] (common order).
        sim_body_pos = self._engine.get_body_pos(dex_id)[env_ids]

        # Target demo skeleton:
        #   wrist_pos [B, 3] + mano_joints [B, (n_body-1)*3] -> [B, n_body, 3]
        tar_wrist = self._demo_data["wrist_pos"][env_ids, seq_idx]
        tar_joints = self._demo_data["mano_joints"][env_ids, seq_idx].reshape(max_envs, -1, 3)
        tar_body_pos = torch.cat([tar_wrist.unsqueeze(1), tar_joints], dim=1)

        line_w = float(self._draw_retarget_skeleton_line_width)
        col_sim = np.array([0.1, 1.0, 0.1, 1.0], dtype=np.float32)
        col_tar = np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32)

        for env_i in range(max_envs):
            sim_np = sim_body_pos[env_i].detach().cpu().numpy()
            tar_np = tar_body_pos[env_i].detach().cpu().numpy()
            n_body = min(sim_np.shape[0], tar_np.shape[0])
            if n_body <= 1:
                continue

            valid = links[(links[:, 0] < n_body) & (links[:, 1] < n_body)]
            if valid.shape[0] == 0:
                continue

            # Sim skeleton (green)
            sim_starts = sim_np[valid[:, 0]]
            sim_ends = sim_np[valid[:, 1]]
            sim_cols = np.repeat(col_sim[None, :], valid.shape[0], axis=0)
            self._engine.draw_lines(env_i, sim_starts, sim_ends, sim_cols, line_w)

            # Demo/retarget skeleton (red)
            tar_starts = tar_np[valid[:, 0]]
            tar_ends = tar_np[valid[:, 1]]
            tar_cols = np.repeat(col_tar[None, :], valid.shape[0], axis=0)
            self._engine.draw_lines(env_i, tar_starts, tar_ends, tar_cols, line_w)
        return

    def _reset_task(self, env_ids):
        """PORT: dexhandimitator._reset_default.
        Set `self._deterministic_reset = True` to skip all reset noise AND use
        opt_dof_pos straight from the demo (instead of default + DoF noise).
        Useful for visual verification tools like tools/frame_sweep.py."""
        dex_id = self._get_dexhand_obj_id()
        n = len(env_ids)
        deterministic = bool(getattr(self, "_deterministic_reset", False))

        # --- pick demo start index ------------------------------------------
        if self._random_state_init:
            seq_idx = torch.floor(
                self._demo_data["seq_len"][env_ids] * 0.99
                * torch.rand_like(self._demo_data["seq_len"][env_ids].float())
            ).long()
        else:
            seq_idx = torch.zeros_like(self._demo_data["seq_len"][env_ids].long())

        # --- DoF pose --------------------------------------------------------
        if deterministic and "opt_dof_pos" in self._demo_data:
            dof_pos = self._demo_data["opt_dof_pos"][env_ids, seq_idx]
            dof_pos = torch.clamp(dof_pos, self._dof_lower, self._dof_upper)
            dof_vel = torch.zeros(n, self._dexhand.n_dofs, device=self._device)
        else:
            default = self._dexhand_default_dof_pos[None].repeat(n, 1)
            if deterministic:
                dof_pos = torch.clamp(default, self._dof_lower, self._dof_upper)
                dof_vel = torch.zeros(n, self._dexhand.n_dofs, device=self._device)
            else:
                noise = torch.randn_like(default) * ((self._dof_upper - self._dof_lower) / 8)[None]
                dof_pos = torch.clamp(default + noise, self._dof_lower, self._dof_upper)
                dof_vel = torch.randn(n, self._dexhand.n_dofs, device=self._device) * 0.1

        # --- wrist pose / vel from demo seq ---------------------------------
        wrist_pos = self._demo_data["wrist_pos"][env_ids, seq_idx].clone()
        if (not deterministic) and (self._reset_wrist_pos_noise_std > 0.0):
            wrist_pos = wrist_pos + torch.randn_like(wrist_pos) * self._reset_wrist_pos_noise_std

        wrist_rot_aa = self._demo_data["wrist_rot"][env_ids, seq_idx]
        wrist_rot = aa_to_rotmat(wrist_rot_aa)
        if (not deterministic) and (self._reset_wrist_rot_noise_rad > 0.0):
            noise_rot = torch.rand(n, 3, device=self._device)
            noise_rot = aa_to_rotmat(
                noise_rot / torch.norm(noise_rot, dim=-1, keepdim=True)
                * torch.randn(n, 1, device=self._device) * self._reset_wrist_rot_noise_rad
            )
            wrist_rot = noise_rot @ wrist_rot
        wrist_rot_q = rotmat_to_quat(wrist_rot)  # wxyz
        wrist_rot_q = wrist_rot_q[:, [1, 2, 3, 0]]  # xyzw (engine convention)

        wrist_vel = self._demo_data["wrist_velocity"][env_ids, seq_idx].clone()
        wrist_ang_vel = self._demo_data["wrist_angular_velocity"][env_ids, seq_idx].clone()
        if (not deterministic) and (self._reset_wrist_vel_noise_std > 0.0):
            wrist_vel = wrist_vel + torch.randn_like(wrist_vel) * self._reset_wrist_vel_noise_std
            wrist_ang_vel = wrist_ang_vel + torch.randn_like(wrist_ang_vel) * self._reset_wrist_vel_noise_std
        else:
            # Static inspection: zero velocity so the hand doesn't drift after pin.
            wrist_vel = torch.zeros_like(wrist_vel)
            wrist_ang_vel = torch.zeros_like(wrist_ang_vel)

        # --- palm-offset-aware reset ---------------------------------------
        # If the USD's articulation root is NOT R_hand_base_link (Isaac Sim's
        # URDF Importer bug with inspire_hand_right_lab.usd picks
        # R_thumb_proximal_base), directly setting root_pose_w with the demo
        # wrist target would place the *articulation root* at that position
        # while the actual palm ends up elsewhere (measured: ~40 cm off).
        palm_sim_idx = self._engine.get_palm_body_sim_idx(dex_id)

        if palm_sim_idx == 0:
            # Simple path: articulation root == palm.
            self._engine.set_root_pos(env_ids, dex_id, wrist_pos)
            self._engine.set_root_rot(env_ids, dex_id, wrist_rot_q)
            self._engine.set_root_vel(env_ids, dex_id, wrist_vel)
            self._engine.set_root_ang_vel(env_ids, dex_id, wrist_ang_vel)
            self._engine.set_dof_pos(env_ids, dex_id, dof_pos)
            self._engine.set_dof_vel(env_ids, dex_id, dof_vel)
        else:
            # Two-phase reset:
            #   (1) Apply DoF target with articulation root pinned at (0, identity)
            #       and flush one sim step so the kinematic chain settles.
            #   (2) Read the palm's actual world pose in that pinned state — this
            #       captures palm offset (both translation and rotation) relative
            #       to the articulation root at the given DoF.
            #   (3) Re-target the articulation root so that the palm ends up at
            #       the demo wrist target.
            identity_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]],
                                         device=self._device).expand(n, 4).contiguous()
            zero_vec = torch.zeros_like(wrist_pos)

            self._engine.set_root_pos(env_ids, dex_id, zero_vec)
            self._engine.set_root_rot(env_ids, dex_id, identity_xyzw)
            self._engine.set_root_vel(env_ids, dex_id, zero_vec)
            self._engine.set_root_ang_vel(env_ids, dex_id, zero_vec)
            self._engine.set_dof_pos(env_ids, dex_id, dof_pos)
            self._engine.set_dof_vel(env_ids, dex_id, dof_vel)

            # Flush resets + advance sim by one control step so palm pose settles.
            self._engine.step()

            # Palm common-ordering index (== self._wrist_body_id from _cache_body_ids).
            palm_com = self._wrist_body_id
            body_pos_all = self._engine.get_body_pos(dex_id)          # [B, n_bodies, 3]
            body_rot_all = self._engine.get_body_rot(dex_id)          # [B, n_bodies, 4] xyzw

            palm_at_id_pos = body_pos_all[env_ids, palm_com]          # [n, 3]
            palm_at_id_rot_xyzw = body_rot_all[env_ids, palm_com]     # [n, 4] xyzw
            palm_at_id_rot_xyzw = palm_at_id_rot_xyzw / (
                torch.norm(palm_at_id_rot_xyzw, dim=-1, keepdim=True) + 1e-8)
            palm_at_id_rot_wxyz = palm_at_id_rot_xyzw[:, [3, 0, 1, 2]]
            palm_at_id_rot_mat = quat_to_rotmat(palm_at_id_rot_wxyz)  # [n, 3, 3]

            # palm_world = root_world @ palm_rel  =>
            # target_root_rot = target_palm_rot @ palm_rel_rot.T
            # target_root_pos = target_palm_pos - target_root_rot @ palm_rel_pos
            palm_rel_rot_inv = palm_at_id_rot_mat.transpose(-1, -2)
            target_root_rot_mat = wrist_rot @ palm_rel_rot_inv                       # [n, 3, 3]
            target_root_pos = wrist_pos - torch.bmm(
                target_root_rot_mat, palm_at_id_pos.unsqueeze(-1)).squeeze(-1)
            target_root_rot_wxyz = rotmat_to_quat(target_root_rot_mat)
            target_root_rot_xyzw = target_root_rot_wxyz[:, [1, 2, 3, 0]]

            self._engine.set_root_pos(env_ids, dex_id, target_root_pos)
            self._engine.set_root_rot(env_ids, dex_id, target_root_rot_xyzw)
            # velocity: use demo wrist vel directly (ignore small cross term from
            # articulation-root angular velocity — good enough for reset).
            self._engine.set_root_vel(env_ids, dex_id, wrist_vel)
            self._engine.set_root_ang_vel(env_ids, dex_id, wrist_ang_vel)

            # ITERATIVE CORRECTION: after one sim step, measure actual palm
            # pose and shift root to cancel the residual. This reduces
            # post-reset wrist error from ~5 cm to <1 cm typically.
            n_iters = 3 if deterministic else 2
            tol = 1e-3  # 1 mm
            for _ in range(n_iters):
                # Re-set DoF each iter so PD actuation doesn't pull it away.
                self._engine.set_dof_pos(env_ids, dex_id, dof_pos)
                self._engine.set_dof_vel(env_ids, dex_id, dof_vel)
                self._engine.set_root_vel(env_ids, dex_id, wrist_vel)
                self._engine.set_root_ang_vel(env_ids, dex_id, wrist_ang_vel)
                self._engine.step()

                body_pos_now = self._engine.get_body_pos(dex_id)
                palm_now = body_pos_now[env_ids, palm_com]
                residual_pos = wrist_pos - palm_now
                if torch.norm(residual_pos, dim=-1).max() < tol:
                    break
                target_root_pos = target_root_pos + residual_pos
                self._engine.set_root_pos(env_ids, dex_id, target_root_pos)

        if self._debug_reset_error:
            self._debug_reset_counter += 1
            if (self._debug_reset_counter % max(1, self._debug_reset_error_interval)) == 0:
                palm_com = self._wrist_body_id
                body_pos_chk = self._engine.get_body_pos(dex_id)[env_ids, palm_com]
                body_rot_chk_xyzw = self._engine.get_body_rot(dex_id)[env_ids, palm_com]
                body_rot_chk_xyzw = body_rot_chk_xyzw / (
                    torch.norm(body_rot_chk_xyzw, dim=-1, keepdim=True) + 1e-8
                )
                body_rot_chk_wxyz = body_rot_chk_xyzw[:, [3, 0, 1, 2]]
                body_rot_chk_mat = quat_to_rotmat(body_rot_chk_wxyz)

                pos_err = torch.norm(wrist_pos - body_pos_chk, dim=-1)
                rel_rot = wrist_rot.transpose(-1, -2) @ body_rot_chk_mat
                tr = rel_rot[:, 0, 0] + rel_rot[:, 1, 1] + rel_rot[:, 2, 2]
                cos_theta = torch.clamp((tr - 1.0) * 0.5, -1.0, 1.0)
                rot_err_deg = torch.rad2deg(torch.acos(cos_theta))
                Logger.print(
                    "[ResetDebug] "
                    f"envs={len(env_ids)} "
                    f"palm_pos_err(mean/max)={pos_err.mean().item():.4f}/{pos_err.max().item():.4f} m "
                    f"palm_rot_err(mean/max)={rot_err_deg.mean().item():.2f}/{rot_err_deg.max().item():.2f} deg "
                    f"deterministic={deterministic}"
                )

        # progress bookkeeping
        self._progress_buf[env_ids] = seq_idx
        self._running_progress_buf[env_ids] = 0
        return
