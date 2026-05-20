"""
Base environment for ManipTrans dexterous manipulation tasks on Isaac Lab.

Plays the same role as MimicKit's char_env.py, but:
  * the "character" is a dexhand (Inspire / Allegro / Shadow / XHand / ArtiMANO),
    loaded through the MimicKit engine abstraction (engine.create_obj).
  * observations are returned as a gymnasium.spaces.Dict with keys
    {"proprioception", "privileged", "target"} — matching ManipTrans's
    DictObs conventions.
  * the first "object" in each env is always the dexhand; subsequent objects
    (table, manipulated prop, visual MANO joint markers) are spawned by
    task subclasses in _build_scene_props().

This file is the Isaac-Lab port of the shared boilerplate that was scattered
across maniptrans_envs/lib/envs/core/vec_task.py (Isaac Gym base) and the
initialization blocks of each dexhandimitator / dexhandmanip_sh / dexhandmanip_bih
task. Task-specific logic (obs, reward, reset) belongs in the subclasses.
"""
import abc
import os

import gymnasium.spaces as spaces
import numpy as np
import torch

import engines.engine as engine
import envs.sim_env as sim_env
import envs.base_env as base_env
import util.camera as camera
from util.logger import Logger

from dexhands.factory import DexHandFactory
from dataset.transform import rot6d_to_aa
from utils.asset_root import ASSET_ROOT  # noqa: F401  (kept for tasks that reference it)


# Default height of the table surface; dexhand root starts this high above the floor.
TABLE_SURFACE_Z = 0.415
ROBOT_HEIGHT = 0.00214874  # copied from ManipTrans/maniptrans_envs/lib/envs/core/config.py


class DexEnv(sim_env.SimEnv):
    """
    Generic dexhand-on-a-table environment.

    Subclasses (DexImitatorEnv, DexManipSHEnv, DexManipBiHEnv) must implement:
      * _build_scene_props(env_id, env_config)   – spawn table/object/markers
      * _build_action_space()                    – per-task action layout
      * _compute_obs(env_ids)                    – dict observation
      * _update_reward()                         – task reward
      * _update_done()                           – termination logic
      * _reset_task(env_ids)                     – task-specific reset (demo seq, obj pose...)
    """

    NAME = "dex_env"

    def __init__(self, env_config, engine_config, num_envs, device,
                 visualize, record_video=False):

        # --- bridge ManipTrans episodeLength (frames) → MimicKit episode_length (sec).
        # MimicKit's SimEnv base expects env_config["episode_length"] in seconds.
        if "episode_length" not in env_config:
            frames = env_config.get("episodeLength", 2000)
            ctrl_freq = engine_config.get("control_freq", 60) if engine_config else 60
            env_config["episode_length"] = float(frames) / float(ctrl_freq)

        # --- read task-level config up front (subclasses may override) -----
        self._side = env_config.get("side", "right")
        self._dexhand_type = env_config["dexhand"]
        self._use_quat_rot = env_config.get("useQuatRot", True)
        self._use_pid_control = env_config.get("usePIDControl", False)
        self._action_scale = env_config.get("actionScale", 1.0)
        self._translation_scale = env_config.get("translationScale", 1.0)
        self._orientation_scale = env_config.get("orientationScale", 0.1)
        self._act_moving_average = env_config.get("actionsMovingAverage", 1.0)
        self._training = env_config.get("training", True)

        # --- instantiate the dexhand wrapper (pure-Python; no sim yet) -----
        self._dexhand = DexHandFactory.create_hand(self._dexhand_type, self._side)

        super().__init__(env_config=env_config,
                         engine_config=engine_config,
                         num_envs=num_envs,
                         device=device,
                         visualize=visualize,
                         record_video=record_video)

        # after engine.initialize_sim() the per-obj ordering is known –
        # cache body handles / contact body ids
        self._cache_body_ids()
        self._build_default_dof_pose()
        return

    # ------------------------------------------------------------------ #
    # Abstract-ish hooks that concrete tasks must override                #
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def _build_scene_props(self, env_id, env_config):
        """Spawn table, manipulation object, MANO visual markers, etc."""
        return

    @abc.abstractmethod
    def _reset_task(self, env_ids):
        """Task-specific reset: sample demo sequence, place object, etc."""
        return

    # ------------------------------------------------------------------ #
    # SimEnv overrides                                                    #
    # ------------------------------------------------------------------ #
    def _build_envs(self, env_config, num_envs):
        self._dexhand_obj_ids = []   # obj_id returned by engine.create_obj for the dexhand
        self._table_obj_ids = []
        self._manip_obj_ids = []     # populated by subclass

        for e in range(num_envs):
            Logger.print("Building {:d}/{:d} envs".format(e + 1, num_envs), end='\r')
            env_id = self._engine.create_env()
            assert env_id == e

            # 1) dexhand — always obj_id 0
            dexhand_start_pos = np.array([-0.4, 0.0, TABLE_SURFACE_Z + ROBOT_HEIGHT])
            dexhand_start_rot = np.array([0.0, -0.70710678, 0.0, 0.70710678])  # (x,y,z,w) = euler_zyx(0, -pi/2, 0)

            # Optional per-joint PD gains. Reference ManipTrans ships Shadow
            # with {stiffness=500, damping=30} and Inspire with similar values;
            # without these set, Isaac Lab falls back to the USD's drive
            # properties (often weaker) and training / replay won't track the
            # target pose as tightly. Format: env yaml's articulation_cfg block.
            art_cfg = env_config.get("articulation_cfg", {}) or {}
            dexhand_stiffness = art_cfg.get("stiffness", None)
            dexhand_damping   = art_cfg.get("damping", None)

            dex_id = self._engine.create_obj(
                env_id=env_id,
                obj_type=engine.ObjType.articulated,
                asset_file=self._dexhand.urdf_path,
                name="dexhand",
                is_visual=False,
                enable_self_collisions=self._dexhand.self_collision,
                fix_root=False,
                start_pos=dexhand_start_pos,
                start_rot=dexhand_start_rot,
                color=np.array([0.2, 0.25, 0.7]),
                disable_motors=False,
                # Matches ManipTrans's asset_options.disable_gravity = True:
                # wrist is *fully* driven by external force/torque from policy.
                # Without this, gravity makes the hand free-fall every step and
                # no wrist-force action can catch up (as seen in debug_rollout).
                disable_gravity=True,
                stiffness=dexhand_stiffness,
                damping=dexhand_damping,
            )
            if e == 0:
                self._dexhand_obj_ids.append(dex_id)
            else:
                assert dex_id == self._dexhand_obj_ids[0]

            # 2) task-specific props
            self._build_scene_props(env_id, env_config)

        Logger.print("\n")
        return

    def _build_sim_tensors(self, env_config):
        super()._build_sim_tensors(env_config)
        num_envs = self.get_num_envs()
        dex_id = self._get_dexhand_obj_id()

        # cache dof limits / pd gains — MimicKit's Engine returns numpy, we keep tensors
        dof_low, dof_high = self._engine.get_obj_dof_limits(0, dex_id)
        self._dof_lower = torch.tensor(dof_low, device=self._device, dtype=torch.float)
        self._dof_upper = torch.tensor(dof_high, device=self._device, dtype=torch.float)

        torque_lim = self._engine.get_obj_torque_limits(0, dex_id)
        self._dof_torque_limits = torch.tensor(torque_lim, device=self._device, dtype=torch.float)

        # action bounds (subclass provides action_space.low / high)
        self._action_bound_low = torch.tensor(self._action_space.low, device=self._device)
        self._action_bound_high = torch.tensor(self._action_space.high, device=self._device)

        # progress bookkeeping (matches ManipTrans's progress_buf / running_progress_buf / success_buf)
        self._progress_buf = torch.zeros(num_envs, device=self._device, dtype=torch.long)
        self._running_progress_buf = torch.zeros(num_envs, device=self._device, dtype=torch.long)
        self._success_buf = torch.zeros(num_envs, device=self._device, dtype=torch.long)
        self._failure_buf = torch.zeros(num_envs, device=self._device, dtype=torch.long)

        # action moving average
        self._prev_targets = torch.zeros((num_envs, self._dexhand.n_dofs),
                                         device=self._device, dtype=torch.float)
        self._curr_targets = torch.zeros_like(self._prev_targets)
        return

    def _build_data_buffers(self):
        """Dict-obs variant of SimEnv._build_data_buffers."""
        num_envs = self.get_num_envs()

        self._reward_buf = torch.zeros(num_envs, device=self._device, dtype=torch.float)
        self._done_buf = torch.zeros(num_envs, device=self._device, dtype=torch.int)
        self._timestep_buf = torch.zeros(num_envs, device=self._device, dtype=torch.int)
        self._time_buf = torch.zeros(num_envs, device=self._device, dtype=torch.float)

        obs_space = self.get_obs_space()
        self._obs_buf = self._alloc_obs_buf(obs_space, num_envs)
        self._info = dict()
        return

    def _alloc_obs_buf(self, obs_space, num_envs):
        if isinstance(obs_space, spaces.Dict):
            return {
                k: torch.zeros([num_envs] + list(sub.shape), device=self._device, dtype=torch.float)
                for k, sub in obs_space.spaces.items()
            }
        else:
            return torch.zeros([num_envs] + list(obs_space.shape), device=self._device, dtype=torch.float)

    def get_obs_space(self):
        """Dexhand envs return a Dict space {proprioception, privileged, target}.
        Subclasses override _obs_dims_dict() to declare per-key widths."""
        dims = self._obs_dims_dict()
        return spaces.Dict({
            k: spaces.Box(low=-np.inf, high=np.inf, shape=(d,), dtype=np.float32)
            for k, d in dims.items() if d > 0
        })

    @abc.abstractmethod
    def _obs_dims_dict(self):
        """Return {'proprioception': int, 'privileged': int, 'target': int}."""
        return {}

    def _update_observations(self, env_ids=None):
        if env_ids is None or len(env_ids) > 0:
            obs = self._compute_obs(env_ids)
            if isinstance(self._obs_buf, dict):
                if env_ids is None:
                    for k, v in obs.items():
                        self._obs_buf[k][:] = v
                else:
                    for k, v in obs.items():
                        self._obs_buf[k][env_ids] = v
            else:
                if env_ids is None:
                    self._obs_buf[:] = obs
                else:
                    self._obs_buf[env_ids] = obs
        return

    # ------------------------------------------------------------------ #
    # Action application                                                  #
    # ------------------------------------------------------------------ #
    def _apply_action(self, actions):
        """
        ManipTrans non-PID action layout (matches the original imitator):
            action[:, 0:3]                 -> wrist force direction   (scaled below)
            action[:, 3:6]                 -> wrist torque direction  (scaled below)
            action[:, 6:6+n_dofs]          -> finger DoF targets

        PID mode inserts 3 extra dims at index 6 (root_ctrl_dim=9), where
        rotation control uses 6D representation converted to axis-angle.
        """
        dex_id = self._get_dexhand_obj_id()

        clip_actions = torch.minimum(
            torch.maximum(actions, self._action_bound_low),
            self._action_bound_high,
        )

        # Same layout decision as ManipTrans's root_control_dim
        root_ctrl_dim = 9 if self._use_pid_control else 6
        dof_action_slice = clip_actions[:, root_ctrl_dim: root_ctrl_dim + self._dexhand.n_dofs]

        # --- DoF position targets (moving-averaged, clamped) --------
        curr = 0.5 * (dof_action_slice + 1.0) * (self._dof_upper - self._dof_lower) + self._dof_lower
        curr = self._act_moving_average * curr + (1.0 - self._act_moving_average) * self._prev_targets
        curr = torch.clamp(curr, self._dof_lower, self._dof_upper)
        self._prev_targets[:] = curr
        self._curr_targets[:] = curr
        self._engine.set_cmd(dex_id, curr)

        # --- Wrist wrench ------------------------------------------
        if self._use_pid_control:
            self._apply_wrist_force_pid(clip_actions)
        else:
            self._apply_wrist_force(clip_actions)
        return

    def _apply_wrist_wrench_with_ma(self, new_force, new_torque):
        """Apply wrist wrench with the same moving-average behavior as ManipTrans."""
        dex_id = self._get_dexhand_obj_id()
        am = self._act_moving_average
        if not hasattr(self, "_wrist_force_buf"):
            nE = self.get_num_envs()
            self._wrist_force_buf = torch.zeros(nE, 3, device=self._device, dtype=torch.float)
            self._wrist_torque_buf = torch.zeros(nE, 3, device=self._device, dtype=torch.float)

        self._wrist_force_buf = am * new_force + (1.0 - am) * self._wrist_force_buf
        self._wrist_torque_buf = am * new_torque + (1.0 - am) * self._wrist_torque_buf

        self._set_body_wrench(
            dex_id, self._wrist_body_id,
            forces=self._wrist_force_buf, torques=self._wrist_torque_buf,
        )
        return

    def _apply_wrist_force(self, clip_actions):
        """Apply an external force + torque to the wrist body so the policy
        can drag the hand along the demo trajectory.

        PORT: dexhandimitator.pre_physics_step (non-PID branch, lines ~960-980 of
        maniptrans_envs/lib/envs/tasks/dexhandimitator.py).
        """
        # _wrist_body_id is cached in _cache_body_ids during __init__.
        dt = self._engine.get_timestep()

        # action[:, 0:3] in [-1, 1] -> force in Newton scale (500 Ns/m²)
        new_force = clip_actions[:, 0:3] * dt * self._translation_scale * 500.0
        new_torque = clip_actions[:, 3:6] * dt * self._orientation_scale * 200.0

        self._apply_wrist_wrench_with_ma(new_force, new_torque)
        return

    def _apply_wrist_force_pid(self, clip_actions):
        """PID wrist control branch (ManipTrans usePIDControl path)."""
        dt = self._engine.get_timestep()
        nE = self.get_num_envs()

        if not hasattr(self, "_pos_error_integral"):
            self._pos_error_integral = torch.zeros(nE, 3, device=self._device, dtype=torch.float)
            self._prev_pos_error = torch.zeros(nE, 3, device=self._device, dtype=torch.float)
            self._rot_error_integral = torch.zeros(nE, 3, device=self._device, dtype=torch.float)
            self._prev_rot_error = torch.zeros(nE, 3, device=self._device, dtype=torch.float)

        position_error = clip_actions[:, 0:3]
        self._pos_error_integral += position_error * dt
        self._pos_error_integral = torch.clamp(self._pos_error_integral, -1.0, 1.0)
        pos_derivative = (position_error - self._prev_pos_error) / dt
        self._prev_pos_error = position_error

        new_force = (
            float(self._dexhand.Kp_pos) * position_error
            + float(self._dexhand.Ki_pos) * self._pos_error_integral
            + float(self._dexhand.Kd_pos) * pos_derivative
        )

        # PID rotation action is 6D representation in ManipTrans.
        rotation_error = rot6d_to_aa(clip_actions[:, 3:9])
        self._rot_error_integral += rotation_error * dt
        self._rot_error_integral = torch.clamp(self._rot_error_integral, -1.0, 1.0)
        rot_derivative = (rotation_error - self._prev_rot_error) / dt
        self._prev_rot_error = rotation_error

        new_torque = (
            float(self._dexhand.Kp_rot) * rotation_error
            + float(self._dexhand.Ki_rot) * self._rot_error_integral
            + float(self._dexhand.Kd_rot) * rot_derivative
        )

        self._apply_wrist_wrench_with_ma(new_force, new_torque)
        return

    def _set_body_wrench(self, obj_id, body_id, forces, torques):
        """One-shot force+torque setter that bypasses MimicKit's set_body_forces
        (which zeros the torque channel). Directly calls the Isaac Lab articulation
        `set_external_force_and_torque` with both wrench components.
        """
        obj = self._engine._objs[obj_id]
        # Isaac Lab expects shape [n_envs, n_bodies_in_call, 3]
        f = forces.unsqueeze(-2)
        t = torques.unsqueeze(-2)
        # body_ids must be a 1-D tensor (0-dim scalar breaks warp.from_torch).
        sim_body_id_scalar = self._engine._body_order_sim2common[obj_id][body_id]
        sim_body_ids = sim_body_id_scalar.reshape(1).to(torch.long)
        obj.set_external_force_and_torque(
            forces=f, torques=t,
            positions=None, env_ids=None,
            body_ids=sim_body_ids, is_global=False,
        )
        # Intentionally DO NOT set obj.has_external_wrentch = True.
        # The MimicKit engine's _clear_forces path uses a [1, 3] tensor that
        # newer Isaac Lab rejects (expects [n_envs, n_bodies, 3]). We manage
        # wrench lifecycle ourselves via the moving-average buffer: each control
        # step we overwrite the applied wrench with the freshly-MA'd value, and
        # if the policy's force action goes to zero the buffer decays to zero.
        return

    # ------------------------------------------------------------------ #
    # Reset                                                               #
    # ------------------------------------------------------------------ #
    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids) > 0:
            # Zero the running/success/failure counters BEFORE _reset_task so that
            # _reset_task can overwrite _progress_buf with its sampled seq_idx
            # (random-state-init) without us clobbering it back to 0 after.
            self._running_progress_buf[env_ids] = 0
            self._success_buf[env_ids] = 0
            self._failure_buf[env_ids] = 0
            self._prev_targets[env_ids] = 0
            self._curr_targets[env_ids] = 0
            # _reset_task:
            #   - teleports wrist/DoF to demo[seq_idx] (+ noise)
            #   - sets _progress_buf[env_ids] = seq_idx
            # so reward/target lookups after this read the same frame the wrist
            # was placed at — eliminating the 22cm reset offset.
            self._reset_task(env_ids)
        return

    def _update_misc(self):
        self._progress_buf += 1
        self._running_progress_buf += 1
        return

    # ------------------------------------------------------------------ #
    # Helpers used by subclasses                                          #
    # ------------------------------------------------------------------ #
    def _get_dexhand_obj_id(self):
        return self._dexhand_obj_ids[0]

    def _cache_body_ids(self):
        dex_id = self._get_dexhand_obj_id()
        self._dexhand_body_ids = {
            name: self._engine.find_obj_body_id(dex_id, name)
            for name in self._dexhand.body_names
        }
        self._contact_body_ids = torch.tensor(
            [self._dexhand_body_ids[k] for k in self._dexhand.contact_body_names],
            device=self._device, dtype=torch.long,
        )
        # Palm body common-ordering index (used by _reset_task palm-offset
        # correction and by _apply_wrist_force for external wrench).
        wrist_dex_name = self._dexhand.to_dex("wrist")[0]
        self._wrist_body_id = self._dexhand_body_ids[wrist_dex_name]
        return

    def _build_default_dof_pose(self):
        default_pose = torch.ones(self._dexhand.n_dofs, device=self._device) * np.pi / 36
        # inspire-specific thumb opening (from ManipTrans DexHandImitator init)
        if self._dexhand_type == "inspire":
            default_pose[8] = 0.3
            default_pose[9] = 0.01
        self._dexhand_default_dof_pos = default_pose
        return

    # ------------------------------------------------------------------ #
    # Camera                                                              #
    # ------------------------------------------------------------------ #
    def _build_camera(self, env_config):
        env_id = 0
        dex_id = self._get_dexhand_obj_id()
        root_pos = self._engine.get_root_pos(dex_id)[env_id].cpu().numpy()

        cam_pos = np.array([root_pos[0] - 1.0, root_pos[1] - 0.8, root_pos[2] + 0.5])
        cam_target = np.array([root_pos[0], root_pos[1], root_pos[2]])

        cam_mode = camera.CameraMode[env_config.get("camera_mode", "still")]
        self._camera = camera.Camera(mode=cam_mode,
                                     engine=self._engine,
                                     pos=cam_pos,
                                     target=cam_target,
                                     track_env_id=env_id,
                                     track_obj_id=dex_id)
        return
