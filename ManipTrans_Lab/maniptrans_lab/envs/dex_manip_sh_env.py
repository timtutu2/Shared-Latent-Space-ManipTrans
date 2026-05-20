"""
Isaac Lab port of
    ManipTrans/maniptrans_envs/lib/envs/tasks/dexhandmanip_sh.py
    (DexHandManipRHEnv / DexHandManipLHEnv — single-hand manipulation)

Extends DexImitatorEnv by:
  * spawning a manipulable rigid object (the 'manip_obj') in each env,
  * enriching privileged obs with {manip_obj_pos, manip_obj_quat,
    manip_obj_vel, manip_obj_ang_vel, manip_obj_com, manip_obj_weight,
    tip_force},
  * adding a manipulation reward that tracks the object's demo trajectory.

Content (obs layout, reset schedule, reward formula) follows ManipTrans's
dexhandmanip_sh.py 1-to-1; Isaac-Gym-specific physics calls are routed
through `self._engine` exactly as in dex_imitator_env.
"""
from __future__ import annotations

import numpy as np
import torch
import gymnasium.spaces as spaces

import engines.engine as engine
import envs.base_env as base_env
import envs.dex_imitator_env as dex_imitator_env
from dataset.transform import aa_to_quat, quat_to_rotmat, rotmat_to_quat, rot6d_to_aa
from utils import torch_jit_utils


class DexManipSHEnv(dex_imitator_env.DexImitatorEnv):
    NAME = "dex_manip_sh"

    def _parse_stage2_randomization_cfg(self, env_config):
        task_cfg = env_config.get("task", {}) or {}
        rand_enabled = bool(task_cfg.get("randomize", False))
        rand_params = task_cfg.get("randomization_params", {}) or {}
        self._rand_frequency = max(int(rand_params.get("frequency", 32)), 1)

        # gravity scaling schedule
        g_cfg = (
            rand_params.get("sim_params", {})
            .get("gravity", {})
        ) or {}
        g_sample = g_cfg.get("external_sample", {}) or {}
        self._gravity_curr_enabled = (
            rand_enabled
            and g_cfg.get("operation", "") == "scaling"
            and g_cfg.get("schedule", "") == "linear_decay"
            and g_sample.get("type", "") == "const_scale"
        )
        self._gravity_init_scale = float(g_sample.get("init_value", 1.0))
        self._gravity_schedule_steps = max(int(g_cfg.get("schedule_steps", 1)), 1)

        # manip object friction scaling schedule
        f_cfg = (
            rand_params.get("actor_params", {})
            .get("manip_obj", {})
            .get("rigid_shape_properties", {})
            .get("friction", {})
        ) or {}
        f_sample = f_cfg.get("external_sample", {}) or {}
        self._friction_curr_enabled = (
            rand_enabled
            and f_cfg.get("operation", "") == "scaling"
            and f_cfg.get("schedule", "") == "linear_decay"
            and f_sample.get("type", "") == "const_scale"
        )
        self._friction_init_scale = float(f_sample.get("init_value", 1.0))
        self._friction_schedule_steps = max(int(f_cfg.get("schedule_steps", 1)), 1)
        self._friction_upper = float(f_sample.get("upper_bound", 1.0e9))
        self._friction_lower = float(f_sample.get("lower_bound", 0.0))

        self._curriculum_enabled = self._gravity_curr_enabled or self._friction_curr_enabled
        self._last_friction_tick = -1
        self._friction_warned = False
        self._manip_obj_default_material_props = None
        return

    def _curriculum_active(self):
        if not self._curriculum_enabled:
            return False
        mode = getattr(self, "_mode", None)
        return mode == base_env.EnvMode.TRAIN

    @staticmethod
    def _linear_decay_scale(init_scale, steps, step):
        alpha = min(max(float(step) / float(max(steps, 1)), 0.0), 1.0)
        return float(init_scale) + (1.0 - float(init_scale)) * alpha

    def _get_curriculum_step(self):
        if not hasattr(self, "_timestep_buf"):
            return 0
        return int(self._timestep_buf.max().item())

    def _apply_gravity_curriculum_force(self):
        if not self._gravity_curr_enabled:
            return
        obj_id = self._get_manip_obj_id()
        n_envs = self.get_num_envs()
        step = self._get_curriculum_step()
        scale = self._linear_decay_scale(self._gravity_init_scale, self._gravity_schedule_steps, step)
        # Effective gravity scaling via compensation force:
        # F_add = m * g * (scale - 1). scale=0 cancels gravity, scale=1 -> zero add.
        add_force = self._manip_obj_mass * self._gravity_vec * (scale - 1.0)
        forces = add_force.view(1, 3).repeat(n_envs, 1).to(device=self._device, dtype=torch.float)
        self._engine.set_body_forces(None, obj_id, 0, forces)
        return

    def _apply_friction_curriculum(self, force=False):
        if (not self._friction_curr_enabled) or (self._manip_obj_default_material_props is None):
            return

        step = self._get_curriculum_step()
        tick = step // self._rand_frequency
        if (not force) and (tick == self._last_friction_tick):
            return

        scale = self._linear_decay_scale(self._friction_init_scale, self._friction_schedule_steps, step)
        mat = self._manip_obj_default_material_props.clone()
        mat[..., 0] = torch.clamp(mat[..., 0] * scale, self._friction_lower, self._friction_upper)
        mat[..., 1] = torch.clamp(mat[..., 1] * scale, self._friction_lower, self._friction_upper)
        try:
            obj_id = self._get_manip_obj_id()
            obj = self._engine._objs[obj_id]
            env_ids_cpu = torch.arange(self.get_num_envs(), dtype=torch.int32, device="cpu")
            obj.root_physx_view.set_material_properties(mat, indices=env_ids_cpu)
            self._last_friction_tick = tick
        except Exception:
            # Keep training running even if backend does not expose this setter.
            if not self._friction_warned:
                print("[DexManipSHEnv] warning: failed to apply friction curriculum; continuing without it.")
                self._friction_warned = True
        return

    def _build_action_space(self):
        """Residual policy action layout matching original ManipTrans.

        Policy outputs residual part only: (wrist force/torque + dof residual).
        Full action [base, residual] is assembled in the rl runner path.
        """
        n_dof = self._dexhand.n_dofs
        action_size = (1 if self._use_quat_rot else 0) + 6 + n_dof
        low = -np.ones(action_size, dtype=np.float32)
        high = np.ones(action_size, dtype=np.float32)
        return spaces.Box(low=low, high=high)

    def _apply_action(self, actions):
        """Residual action application (base + delta) ported from ManipTrans."""
        dex_id = self._get_dexhand_obj_id()
        # Full action comes from rl runner as [base_action, residual_action].
        clip_actions = torch.clamp(actions, -1.0, 1.0)

        n_dof = self._dexhand.n_dofs
        root_ctrl_dim = 9 if self._use_pid_control else 6
        if not self._use_pid_control:
            res_split_idx = clip_actions.shape[1] // 2
        else:
            res_split_idx = ((clip_actions.shape[1] - (root_ctrl_dim - 6)) // 2 + (root_ctrl_dim - 6))

        base_action = clip_actions[:, :res_split_idx]
        residual_action = clip_actions[:, res_split_idx:] * 2.0

        dof_pos = (
            base_action[:, root_ctrl_dim: root_ctrl_dim + n_dof]
            + residual_action[:, 6: 6 + n_dof]
        )
        dof_pos = torch.clamp(dof_pos, -1.0, 1.0)

        curr = 0.5 * (dof_pos + 1.0) * (self._dof_upper - self._dof_lower) + self._dof_lower
        curr = self._act_moving_average * curr + (1.0 - self._act_moving_average) * self._prev_targets
        curr = torch.clamp(curr, self._dof_lower, self._dof_upper)
        self._prev_targets[:] = curr
        self._curr_targets[:] = curr
        self._engine.set_cmd(dex_id, curr)

        dt = self._engine.get_timestep()
        if self._use_pid_control:
            nE = self.get_num_envs()
            if not hasattr(self, "_pos_error_integral"):
                self._pos_error_integral = torch.zeros(nE, 3, device=self._device, dtype=torch.float)
                self._prev_pos_error = torch.zeros(nE, 3, device=self._device, dtype=torch.float)
                self._rot_error_integral = torch.zeros(nE, 3, device=self._device, dtype=torch.float)
                self._prev_rot_error = torch.zeros(nE, 3, device=self._device, dtype=torch.float)

            position_error = base_action[:, 0:3]
            self._pos_error_integral += position_error * dt
            self._pos_error_integral = torch.clamp(self._pos_error_integral, -1.0, 1.0)
            pos_derivative = (position_error - self._prev_pos_error) / dt
            self._prev_pos_error = position_error

            force = (
                float(self._dexhand.Kp_pos) * position_error
                + float(self._dexhand.Ki_pos) * self._pos_error_integral
                + float(self._dexhand.Kd_pos) * pos_derivative
            )
            force = force + residual_action[:, 0:3] * dt * self._translation_scale * 500.0

            rotation_error = rot6d_to_aa(base_action[:, 3:root_ctrl_dim])
            self._rot_error_integral += rotation_error * dt
            self._rot_error_integral = torch.clamp(self._rot_error_integral, -1.0, 1.0)
            rot_derivative = (rotation_error - self._prev_rot_error) / dt
            self._prev_rot_error = rotation_error

            torque = (
                float(self._dexhand.Kp_rot) * rotation_error
                + float(self._dexhand.Ki_rot) * self._rot_error_integral
                + float(self._dexhand.Kd_rot) * rot_derivative
            )
            torque = torque + residual_action[:, 3:6] * dt * self._orientation_scale * 200.0
        else:
            force = (
                base_action[:, 0:3] * dt * self._translation_scale * 500.0
                + residual_action[:, 0:3] * dt * self._translation_scale * 500.0
            )
            torque = (
                base_action[:, 3:6] * dt * self._orientation_scale * 200.0
                + residual_action[:, 3:6] * dt * self._orientation_scale * 200.0
            )

        self._apply_wrist_wrench_with_ma(force, torque)
        if self._curriculum_active():
            self._apply_gravity_curriculum_force()
        return

    def _ensure_tip_contact_history(self):
        if hasattr(self, "_tips_contact_history"):
            return
        contact_hist_len = 3  # ManipTrans dexhandmanip_sh.py: CONTACT_HISTORY_LEN
        num_contacts = len(self._dexhand.contact_body_names)
        self._tips_contact_history = torch.ones(
            self.get_num_envs(), contact_hist_len, num_contacts,
            device=self._device, dtype=torch.bool,
        )
        return

    def __init__(self, env_config, engine_config, num_envs, device,
                 visualize, record_video=False):
        self._manip_obj_asset = env_config["manipObjAsset"]  # path to USD / URDF
        # Match ManipTrans Gym Stage-2: append BPS(128) to target obs.
        self._use_bps_in_target = bool(env_config.get("useBPSInTarget", True))
        self._bps_dim = int(env_config.get("bpsDim", 128))
        self._obj_bps = None
        self._zero_bps = None
        self._parse_stage2_randomization_cfg(env_config)
        self._priv_obs_keys = env_config.get("privilegedObsKeys", [
            "dq", "manip_obj_pos", "manip_obj_quat",
            "manip_obj_vel", "manip_obj_ang_vel",
            "tip_force", "manip_obj_com", "manip_obj_weight",
        ])
        super().__init__(env_config=env_config,
                         engine_config=engine_config,
                         num_envs=num_envs,
                         device=device,
                         visualize=visualize,
                         record_video=record_video)
        self._init_bps_features()
        self._ensure_tip_contact_history()
        return

    def _init_bps_features(self):
        if not self._use_bps_in_target:
            return
        num_envs = self.get_num_envs()
        self._zero_bps = torch.zeros((num_envs, self._bps_dim), device=self._device, dtype=torch.float)
        obj_verts = self._demo_data.get("obj_verts", None)
        if not isinstance(obj_verts, torch.Tensor) or obj_verts.ndim != 3:
            print("[DexManipSHEnv] warning: obj_verts missing; using zero BPS features.")
            self._obj_bps = self._zero_bps
            return
        try:
            from bps_torch.bps import bps_torch
            bps_layer = bps_torch(
                bps_type="grid_sphere",
                n_bps_points=self._bps_dim,
                radius=0.2,
                randomize=False,
                device=self._device,
            )
            feat = bps_layer.encode(obj_verts, feature_type="dists")["dists"]
            feat = feat.to(device=self._device, dtype=torch.float)
            if feat.shape[-1] != self._bps_dim:
                print("[DexManipSHEnv] warning: BPS dim mismatch; using zero BPS features.")
                self._obj_bps = self._zero_bps
            else:
                self._obj_bps = feat
        except Exception as e:
            print(f"[DexManipSHEnv] warning: failed to build BPS features ({e}); using zeros.")
            self._obj_bps = self._zero_bps
        return

    def _build_sim_tensors(self, env_config):
        super()._build_sim_tensors(env_config)
        obj_id = self._get_manip_obj_id()
        self._manip_obj_mass = float(self._engine.calc_obj_mass(0, obj_id))
        self._gravity_vec = torch.tensor(self._engine.get_gravity(), device=self._device, dtype=torch.float)

        if self._friction_curr_enabled:
            try:
                obj = self._engine._objs[obj_id]
                self._manip_obj_default_material_props = (
                    obj.root_physx_view.get_material_properties().clone().to("cpu").float()
                )
                self._apply_friction_curriculum(force=True)
            except Exception:
                self._manip_obj_default_material_props = None
                if not self._friction_warned:
                    print("[DexManipSHEnv] warning: failed to initialize friction curriculum; continuing without it.")
                    self._friction_warned = True
        return

    # ------------------------------------------------------------------ #
    # Scene                                                               #
    # ------------------------------------------------------------------ #
    def _build_scene_props(self, env_id, env_config):
        # table first (reuse imitator logic)
        super()._build_scene_props(env_id, env_config)

        # manip object ---------------------------------------------------
        start_pos = np.array(env_config.get("manipObjInitPos", [0.0, 0.0, 0.5]))
        start_rot = np.array(env_config.get("manipObjInitRot", [0.0, 0.0, 0.0, 1.0]))
        obj_id = self._engine.create_obj(
            env_id=env_id,
            obj_type=engine.ObjType.rigid,
            asset_file=self._manip_obj_asset,
            name="manip_obj",
            is_visual=False,
            fix_root=False,
            start_pos=start_pos,
            start_rot=start_rot,
            color=np.array([0.8, 0.5, 0.2]),
        )
        if env_id == 0:
            self._manip_obj_ids.append(obj_id)
        return

    def _get_manip_obj_id(self):
        return self._manip_obj_ids[0]

    # ------------------------------------------------------------------ #
    # Observation                                                         #
    # ------------------------------------------------------------------ #
    def _obs_dims_dict(self):
        dims = super()._obs_dims_dict()

        # augment privileged obs
        extra = 0
        if "manip_obj_pos" in self._priv_obs_keys: extra += 3
        if "manip_obj_quat" in self._priv_obs_keys: extra += 4
        if "manip_obj_vel" in self._priv_obs_keys: extra += 3
        if "manip_obj_ang_vel" in self._priv_obs_keys: extra += 3
        if "manip_obj_com" in self._priv_obs_keys: extra += 3
        if "manip_obj_weight" in self._priv_obs_keys: extra += 1
        if "tip_force" in self._priv_obs_keys:
            extra += 4 * len(self._dexhand.contact_body_names)  # xyz + magnitude
        dims["privileged"] = dims.get("privileged", 0) + extra
        if self._use_bps_in_target:
            dims["target"] = dims.get("target", 0) + self._bps_dim
        return dims

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)

        dex_id = self._get_dexhand_obj_id()
        obj_id = self._get_manip_obj_id()

        priv_list = []
        if "dq" in self._priv_obs_keys:
            priv_list.append(self._engine.get_dof_vel(dex_id))
        if "manip_obj_pos" in self._priv_obs_keys:
            pos = self._engine.get_root_pos(obj_id) - self._engine.get_root_pos(dex_id)
            priv_list.append(pos)
        if "manip_obj_quat" in self._priv_obs_keys:
            priv_list.append(self._engine.get_root_rot(obj_id))
        if "manip_obj_vel" in self._priv_obs_keys:
            priv_list.append(self._engine.get_root_vel(obj_id))
        if "manip_obj_ang_vel" in self._priv_obs_keys:
            priv_list.append(self._engine.get_root_ang_vel(obj_id))
        if "manip_obj_com" in self._priv_obs_keys:
            # approximate: COM ≈ root pos
            com = self._engine.get_root_pos(obj_id) - self._engine.get_root_pos(dex_id)
            priv_list.append(com)
        if "manip_obj_weight" in self._priv_obs_keys:
            mass = self._engine.calc_obj_mass(0, obj_id)
            # Match ManipTrans dexhandmanip_sh.py:661 — weight obs uses the
            # sim's actual gravity magnitude (z component), not a hardcoded 9.81.
            gz = self._engine.get_gravity()[2]
            w = torch.full((self.get_num_envs(), 1), float(mass) * -float(gz),
                           device=self._device)
            priv_list.append(w)
        if "tip_force" in self._priv_obs_keys:
            contact = self._engine.get_contact_forces(dex_id)  # [B, n_bodies, 3]
            tips = contact[:, self._contact_body_ids, :]
            mag = torch.norm(tips, dim=-1, keepdim=True)
            priv_list.append(torch.cat([tips, mag], dim=-1).reshape(self.get_num_envs(), -1))

        priv = torch.cat(priv_list, dim=-1)
        if env_ids is not None:
            priv = priv[env_ids] if priv.shape[0] == self.get_num_envs() else priv
        obs["privileged"] = priv

        if self._use_bps_in_target:
            if isinstance(self._obj_bps, torch.Tensor):
                bps = self._obj_bps
            else:
                if self._zero_bps is None or self._zero_bps.shape[0] != self.get_num_envs():
                    self._zero_bps = torch.zeros((self.get_num_envs(), self._bps_dim), device=self._device, dtype=torch.float)
                bps = self._zero_bps
            if env_ids is not None:
                bps = bps[env_ids]
            obs["target"] = torch.cat([obs["target"], bps], dim=-1)
        return obs

    # ------------------------------------------------------------------ #
    # Reward — override to include object tracking                        #
    # ------------------------------------------------------------------ #
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

    def _update_misc(self):
        super()._update_misc()
        if self._curriculum_active():
            self._apply_friction_curriculum()
        return

    def _update_reward(self):
        """PORT of ManipTrans dexhandmanip_sh.compute_reward / compute_imitation_reward."""
        self._ensure_tip_contact_history()
        dex_id = self._get_dexhand_obj_id()
        obj_id = self._get_manip_obj_id()
        nE = self.get_num_envs()
        bs = torch.arange(nE, device=self._device)
        # Clamp by per-env sequence length to avoid OOB indexing when episodes
        # continue a few steps past the last demo frame.
        cur_idx = torch.minimum(self._progress_buf, self._demo_data["seq_len"] - 1)
        max_length = torch.clamp(self._demo_data["seq_len"], 0, self._max_episode_length).float()

        q = self._engine.get_dof_pos(dex_id)
        dq = self._engine.get_dof_vel(dex_id)
        root_pos = self._engine.get_root_pos(dex_id)
        root_rot = self._engine.get_root_rot(dex_id)
        root_vel = self._engine.get_root_vel(dex_id)
        root_ang = self._engine.get_root_ang_vel(dex_id)
        base_state = torch.cat([root_pos, root_rot, root_vel, root_ang], dim=-1)

        body_pos = self._engine.get_body_pos(dex_id)
        body_rot = self._engine.get_body_rot(dex_id)
        body_vel = self._engine.get_body_vel(dex_id)
        body_ang = self._engine.get_body_ang_vel(dex_id)
        joints_state = torch.cat([body_pos, body_rot, body_vel, body_ang], dim=-1)

        obj_pos = self._engine.get_root_pos(obj_id)
        obj_quat = self._engine.get_root_rot(obj_id)
        obj_vel = self._engine.get_root_vel(obj_id)
        obj_ang = self._engine.get_root_ang_vel(obj_id)

        tar_wrist_pos = self._demo_data["wrist_pos"][bs, cur_idx]
        tar_wrist_quat = aa_to_quat(self._demo_data["wrist_rot"][bs, cur_idx])[:, [1, 2, 3, 0]]
        tar_wrist_vel = self._demo_data["wrist_velocity"][bs, cur_idx]
        tar_wrist_ang = self._demo_data["wrist_angular_velocity"][bs, cur_idx]
        tar_joints_pos = self._demo_data["mano_joints"][bs, cur_idx].reshape(nE, -1, 3)
        tar_joints_vel = self._demo_data["mano_joints_velocity"][bs, cur_idx].reshape(nE, -1, 3)

        tar_obj_pose = self._demo_data["obj_trajectory"][bs, cur_idx]
        tar_obj_pos = tar_obj_pose[:, :3, 3]
        tar_obj_quat = rotmat_to_quat(tar_obj_pose[:, :3, :3])[:, [1, 2, 3, 0]]
        tar_obj_vel = self._demo_data["obj_velocity"][bs, cur_idx]
        tar_obj_ang = self._demo_data["obj_angular_velocity"][bs, cur_idx]

        tip_force = self._engine.get_contact_forces(dex_id)[:, self._contact_body_ids, :]
        self._tips_contact_history = torch.cat(
            [
                self._tips_contact_history[:, 1:],
                (torch.norm(tip_force, dim=-1) > 0)[:, None],
            ],
            dim=1,
        )

        dof_torque = self._engine.get_dof_forces(dex_id)
        power = torch.abs(dof_torque * dq).sum(dim=-1)
        if hasattr(self, "_wrist_force_buf"):
            wrist_power = torch.abs(torch.sum(self._wrist_force_buf * root_vel, dim=-1))
            wrist_power += torch.abs(torch.sum(self._wrist_torque_buf * root_ang, dim=-1))
        else:
            wrist_power = torch.zeros(nE, device=self._device, dtype=torch.float)

        diff_eef_pos_dist = torch.norm(tar_wrist_pos - root_pos, dim=-1)
        diff_eef_vel = tar_wrist_vel - root_vel
        diff_eef_ang = tar_wrist_ang - root_ang

        joints_pos = joints_state[:, 1:, :3]
        joints_vel = joints_state[:, 1:, 7:10]
        diff_joints_pos_dist = torch.norm(tar_joints_pos - joints_pos, dim=-1)
        diff_joints_vel = tar_joints_vel - joints_vel

        wi = self._dexhand.weight_idx
        def sel(name):
            ids = [k - 1 for k in wi[name]]
            return diff_joints_pos_dist[:, ids].mean(dim=-1)

        d_thumb = sel("thumb_tip")
        d_index = sel("index_tip")
        d_middle = sel("middle_tip")
        d_ring = sel("ring_tip")
        d_pinky = sel("pinky_tip")
        d_lvl1 = sel("level_1_joints")
        d_lvl2 = sel("level_2_joints")

        r_eef_pos = torch.exp(-40 * diff_eef_pos_dist)
        r_thumb = torch.exp(-100 * d_thumb)
        r_index = torch.exp(-90 * d_index)
        r_middle = torch.exp(-80 * d_middle)
        r_pinky = torch.exp(-60 * d_pinky)
        r_ring = torch.exp(-60 * d_ring)
        r_lvl1 = torch.exp(-50 * d_lvl1)
        r_lvl2 = torch.exp(-40 * d_lvl2)
        r_eef_vel = torch.exp(-1 * diff_eef_vel.abs().mean(dim=-1))
        r_eef_ang = torch.exp(-1 * diff_eef_ang.abs().mean(dim=-1))
        r_joints_vel = torch.exp(-1 * diff_joints_vel.abs().mean(dim=-1).mean(dim=-1))

        diff_eef_rot = torch_jit_utils.quat_mul(
            tar_wrist_quat, torch_jit_utils.quat_conjugate(root_rot)
        )
        diff_eef_rot_angle, _ = torch_jit_utils.quat_to_angle_axis(diff_eef_rot)
        r_eef_rot = torch.exp(-1 * diff_eef_rot_angle.abs())

        diff_obj_pos_dist = torch.norm(tar_obj_pos - obj_pos, dim=-1)
        r_obj_pos = torch.exp(-80 * diff_obj_pos_dist)

        diff_obj_rot = torch_jit_utils.quat_mul(
            tar_obj_quat, torch_jit_utils.quat_conjugate(obj_quat)
        )
        diff_obj_rot_angle, _ = torch_jit_utils.quat_to_angle_axis(diff_obj_rot)
        r_obj_rot = torch.exp(-3 * diff_obj_rot_angle.abs())

        diff_obj_vel = tar_obj_vel - obj_vel
        diff_obj_ang = tar_obj_ang - obj_ang
        r_obj_vel = torch.exp(-1 * diff_obj_vel.abs().mean(dim=-1))
        r_obj_ang = torch.exp(-1 * diff_obj_ang.abs().mean(dim=-1))

        r_power = torch.exp(-10 * power)
        r_wrist_power = torch.exp(-2 * wrist_power)

        tip_dist = self._demo_data["tips_distance"][bs, cur_idx]
        contact_range = [0.02, 0.03]
        tip_weight = torch.clamp(
            (contact_range[1] - tip_dist) / (contact_range[1] - contact_range[0]), 0, 1
        )
        tip_force_masked = tip_force * tip_weight[:, :, None]
        r_tip_force = torch.exp(
            -1 * (1 / (torch.norm(tip_force_masked, dim=-1).sum(dim=-1) + 1e-5))
        )

        reward = (
            0.1 * r_eef_pos
            + 0.6 * r_eef_rot
            + 0.9 * r_thumb
            + 0.8 * r_index
            + 0.75 * r_middle
            + 0.6 * r_pinky
            + 0.6 * r_ring
            + 0.5 * r_lvl1
            + 0.3 * r_lvl2
            + 5.0 * r_obj_pos
            + 1.0 * r_obj_rot
            + 0.1 * r_eef_vel
            + 0.05 * r_eef_ang
            + 0.1 * r_joints_vel
            + 0.1 * r_obj_vel
            + 0.1 * r_obj_ang
            + 1.0 * r_tip_force
            + 0.5 * r_power
            + 0.5 * r_wrist_power
        )

        error_buf = (
            (torch.norm(root_vel, dim=-1) > 100)
            | (torch.norm(root_ang, dim=-1) > 200)
            | (torch.norm(joints_vel, dim=-1).mean(dim=-1) > 100)
            | (torch.abs(dq).mean(dim=-1) > 200)
            | (torch.norm(obj_vel, dim=-1) > 100)
            | (torch.norm(obj_ang, dim=-1) > 200)
        )

        scale_factor = self._compute_tighten_scale_factor()
        failed = (
            (
                (diff_obj_pos_dist > 0.02 / 0.343 * scale_factor ** 3)
                | (d_thumb > 0.04 / 0.7 * scale_factor)
                | (d_index > 0.045 / 0.7 * scale_factor)
                | (d_middle > 0.05 / 0.7 * scale_factor)
                | (d_pinky > 0.06 / 0.7 * scale_factor)
                | (d_ring > 0.06 / 0.7 * scale_factor)
                | (d_lvl1 > 0.07 / 0.7 * scale_factor)
                | (d_lvl2 > 0.08 / 0.7 * scale_factor)
                | (diff_obj_rot_angle.abs() / np.pi * 180 > 30 / 0.343 * scale_factor ** 3)
                | torch.any((tip_dist < 0.005) & ~(self._tips_contact_history.any(dim=1)), dim=-1)
            )
            & (self._running_progress_buf >= 8)
        ) | error_buf

        reached_end = (self._progress_buf + 1 + 3 >= max_length)
        succeeded = reached_end & ~failed

        self._reward_buf[:] = reward
        self._failure_buf[:] = failed.long()
        self._success_buf[:] = succeeded.long()
        return

    # ------------------------------------------------------------------ #
    # Reset — also spawn object at demo start pose                        #
    # ------------------------------------------------------------------ #
    def _reset_task(self, env_ids):
        self._ensure_tip_contact_history()
        dex_id = self._get_dexhand_obj_id()
        obj_id = self._get_manip_obj_id()
        n = len(env_ids)

        if self._random_state_init:
            seq_idx = torch.floor(
                self._demo_data["seq_len"][env_ids] * 0.98
                * torch.rand_like(self._demo_data["seq_len"][env_ids].float())
            ).long()
        else:
            seq_idx = torch.zeros_like(self._demo_data["seq_len"][env_ids].long())

        if "opt_dof_pos" not in self._demo_data or "opt_wrist_pos" not in self._demo_data:
            super()._reset_task(env_ids)
            seq_idx = self._progress_buf[env_ids]
        else:
            dof_pos = torch.clamp(
                self._demo_data["opt_dof_pos"][env_ids, seq_idx],
                self._dof_lower,
                self._dof_upper,
            )
            dof_vel = self._demo_data.get(
                "opt_dof_velocity",
                torch.zeros_like(self._demo_data["opt_dof_pos"]),
            )[env_ids, seq_idx]

            wrist_pos = self._demo_data["opt_wrist_pos"][env_ids, seq_idx]
            wrist_rot = aa_to_quat(self._demo_data["opt_wrist_rot"][env_ids, seq_idx])[:, [1, 2, 3, 0]]
            wrist_vel = self._demo_data.get(
                "opt_wrist_velocity",
                torch.zeros_like(self._demo_data["wrist_velocity"]),
            )[env_ids, seq_idx]
            wrist_ang_vel = self._demo_data.get(
                "opt_wrist_angular_velocity",
                torch.zeros_like(self._demo_data["wrist_angular_velocity"]),
            )[env_ids, seq_idx]

            palm_sim_idx = self._engine.get_palm_body_sim_idx(dex_id)
            if palm_sim_idx == 0:
                self._engine.set_root_pos(env_ids, dex_id, wrist_pos)
                self._engine.set_root_rot(env_ids, dex_id, wrist_rot)
                self._engine.set_root_vel(env_ids, dex_id, wrist_vel)
                self._engine.set_root_ang_vel(env_ids, dex_id, wrist_ang_vel)
                self._engine.set_dof_pos(env_ids, dex_id, dof_pos)
                self._engine.set_dof_vel(env_ids, dex_id, dof_vel)
            else:
                zero_vec = torch.zeros_like(wrist_pos)
                identity_xyzw = torch.tensor(
                    [[0.0, 0.0, 0.0, 1.0]], device=self._device
                ).expand(n, 4).contiguous()

                self._engine.set_root_pos(env_ids, dex_id, zero_vec)
                self._engine.set_root_rot(env_ids, dex_id, identity_xyzw)
                self._engine.set_root_vel(env_ids, dex_id, zero_vec)
                self._engine.set_root_ang_vel(env_ids, dex_id, zero_vec)
                self._engine.set_dof_pos(env_ids, dex_id, dof_pos)
                self._engine.set_dof_vel(env_ids, dex_id, dof_vel)
                self._engine.step()

                palm_com = self._wrist_body_id
                body_pos_all = self._engine.get_body_pos(dex_id)
                body_rot_all = self._engine.get_body_rot(dex_id)
                palm_at_id_pos = body_pos_all[env_ids, palm_com]
                palm_at_id_rot_xyzw = body_rot_all[env_ids, palm_com]
                palm_at_id_rot_xyzw = palm_at_id_rot_xyzw / (
                    torch.norm(palm_at_id_rot_xyzw, dim=-1, keepdim=True) + 1e-8
                )
                palm_at_id_rot_mat = quat_to_rotmat(palm_at_id_rot_xyzw[:, [3, 0, 1, 2]])
                target_root_rot_mat = quat_to_rotmat(wrist_rot[:, [3, 0, 1, 2]]) @ palm_at_id_rot_mat.transpose(-1, -2)
                target_root_pos = wrist_pos - torch.bmm(
                    target_root_rot_mat, palm_at_id_pos.unsqueeze(-1)
                ).squeeze(-1)
                target_root_rot_xyzw = rotmat_to_quat(target_root_rot_mat)[:, [1, 2, 3, 0]]

                self._engine.set_root_pos(env_ids, dex_id, target_root_pos)
                self._engine.set_root_rot(env_ids, dex_id, target_root_rot_xyzw)
                self._engine.set_root_vel(env_ids, dex_id, wrist_vel)
                self._engine.set_root_ang_vel(env_ids, dex_id, wrist_ang_vel)

                for _ in range(2):
                    self._engine.set_dof_pos(env_ids, dex_id, dof_pos)
                    self._engine.set_dof_vel(env_ids, dex_id, dof_vel)
                    self._engine.set_root_vel(env_ids, dex_id, wrist_vel)
                    self._engine.set_root_ang_vel(env_ids, dex_id, wrist_ang_vel)
                    self._engine.step()
                    palm_now = self._engine.get_body_pos(dex_id)[env_ids, palm_com]
                    residual = wrist_pos - palm_now
                    target_root_pos = target_root_pos + residual
                    self._engine.set_root_pos(env_ids, dex_id, target_root_pos)

            self._progress_buf[env_ids] = seq_idx
            self._running_progress_buf[env_ids] = 0

        pose = self._demo_data["obj_trajectory"][env_ids, seq_idx]
        pos = pose[:, :3, 3]
        rot_xyzw = rotmat_to_quat(pose[:, :3, :3])[:, [1, 2, 3, 0]]
        obj_vel = self._demo_data["obj_velocity"][env_ids, seq_idx]
        obj_ang_vel = self._demo_data["obj_angular_velocity"][env_ids, seq_idx]
        self._engine.set_root_pos(env_ids, obj_id, pos)
        self._engine.set_root_rot(env_ids, obj_id, rot_xyzw)
        self._engine.set_root_vel(env_ids, obj_id, obj_vel)
        self._engine.set_root_ang_vel(env_ids, obj_id, obj_ang_vel)
        self._tips_contact_history[env_ids] = True
        return
