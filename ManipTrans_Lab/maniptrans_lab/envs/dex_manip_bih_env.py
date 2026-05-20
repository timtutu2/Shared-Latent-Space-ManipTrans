"""
Isaac Lab port of
    ManipTrans/maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py
    (DexHandManipBiHEnv — bimanual manipulation)

Bimanual variant: two dexhands (right + left) cooperate on a single object.
Structurally it is DexManipSHEnv with a second hand and a symmetric reward.

Observations become double-sized (concat of RH and LH proprio + shared
privileged obs on the manip object). Action space also doubles.
"""
from __future__ import annotations

import numpy as np
import torch

import gymnasium.spaces as spaces

import engines.engine as engine
import envs.dex_manip_sh_env as dex_manip_sh_env
from dexhands.factory import DexHandFactory
from dataset.transform import rotmat_to_quat


class DexManipBiHEnv(dex_manip_sh_env.DexManipSHEnv):
    """Two-hand manipulation env.

    Hand layout (engine obj ordering):
        obj 0: right dexhand (inherited from DexEnv)
        obj 1: table          (inherited from DexImitatorEnv)
        obj 2: manip object   (inherited from DexManipSHEnv)
        obj 3: left dexhand   (added here)
    """

    NAME = "dex_manip_bih"

    def __init__(self, env_config, engine_config, num_envs, device,
                 visualize, record_video=False):
        # left-hand dexhand wrapper; right is built by DexEnv via _side="right"
        self._side = "right"  # primary hand
        self._left_dexhand = DexHandFactory.create_hand(env_config["dexhand"], "left")

        super().__init__(env_config=env_config,
                         engine_config=engine_config,
                         num_envs=num_envs,
                         device=device,
                         visualize=visualize,
                         record_video=record_video)

        # after super init, cache the LH body ids
        self._lh_body_ids = {
            name: self._engine.find_obj_body_id(self._get_lh_dexhand_obj_id(), name)
            for name in self._left_dexhand.body_names
        }
        return

    # ------------------------------------------------------------------ #
    # Scene                                                               #
    # ------------------------------------------------------------------ #
    def _build_scene_props(self, env_id, env_config):
        # table + manip object come from parent
        super()._build_scene_props(env_id, env_config)

        # left dexhand -------------------------------------------------
        import envs.dex_env as dex_env
        start_pos = np.array([-0.4, 0.2, dex_env.TABLE_SURFACE_Z + dex_env.ROBOT_HEIGHT])
        start_rot = np.array([0.0, -0.70710678, 0.0, 0.70710678])

        lh_id = self._engine.create_obj(
            env_id=env_id,
            obj_type=engine.ObjType.articulated,
            asset_file=self._left_dexhand.urdf_path,
            name="dexhand_lh",
            is_visual=False,
            enable_self_collisions=self._left_dexhand.self_collision,
            fix_root=False,
            start_pos=start_pos,
            start_rot=start_rot,
            color=np.array([0.7, 0.25, 0.2]),
            disable_motors=False,
        )
        if env_id == 0:
            self._lh_obj_ids = [lh_id]
        return

    def _get_lh_dexhand_obj_id(self):
        return self._lh_obj_ids[0]

    # ------------------------------------------------------------------ #
    # Obs / Action space — both hands                                    #
    # ------------------------------------------------------------------ #
    def _obs_dims_dict(self):
        dims = super()._obs_dims_dict()
        # proprioception doubles (RH + LH concatenated)
        n_dof = self._dexhand.n_dofs
        lh_prop = 3 * n_dof + 13  # same layout as RH
        dims["proprioception"] = dims["proprioception"] + lh_prop
        return dims

    def _build_action_space(self):
        rh_space = super()._build_action_space()
        action_size = int(np.prod(rh_space.shape)) * 2  # double for LH
        low = -np.ones(action_size, dtype=np.float32)
        high = np.ones(action_size, dtype=np.float32)
        return spaces.Box(low=low, high=high)

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)
        lh_id = self._get_lh_dexhand_obj_id()

        q = self._engine.get_dof_pos(lh_id)
        root_pos = self._engine.get_root_pos(lh_id)
        root_rot = self._engine.get_root_rot(lh_id)
        root_vel = self._engine.get_root_vel(lh_id)
        root_ang_vel = self._engine.get_root_ang_vel(lh_id)
        zero_pos = torch.zeros_like(root_pos)
        lh_base_state = torch.cat([zero_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        lh_prop = torch.cat([q, torch.cos(q), torch.sin(q), lh_base_state], dim=-1)

        if env_ids is not None and lh_prop.shape[0] == self.get_num_envs():
            lh_prop = lh_prop[env_ids]

        obs["proprioception"] = torch.cat([obs["proprioception"], lh_prop], dim=-1)
        return obs

    # ------------------------------------------------------------------ #
    # Actions — split between RH and LH                                   #
    # ------------------------------------------------------------------ #
    def _apply_action(self, actions):
        half = actions.shape[-1] // 2
        rh_actions = actions[:, :half]
        lh_actions = actions[:, half:]

        # delegate RH to base class
        super()._apply_action(rh_actions)

        # symmetric logic for LH
        lh_id = self._get_lh_dexhand_obj_id()
        clip = torch.minimum(torch.maximum(lh_actions, self._action_bound_low[:half]),
                             self._action_bound_high[:half])
        root_ctrl_dim = (1 if self._use_quat_rot else 0) + 6 + (3 if self._use_pid_control else 0)
        dof_slice = clip[:, root_ctrl_dim: root_ctrl_dim + self._left_dexhand.n_dofs]
        curr = 0.5 * (dof_slice + 1.0) * (self._dof_upper - self._dof_lower) + self._dof_lower
        curr = torch.clamp(curr, self._dof_lower, self._dof_upper)
        self._engine.set_cmd(lh_id, curr)
        return

    # ------------------------------------------------------------------ #
    # Reset — also reset LH                                               #
    # ------------------------------------------------------------------ #
    def _reset_task(self, env_ids):
        super()._reset_task(env_ids)
        lh_id = self._get_lh_dexhand_obj_id()
        n = len(env_ids)

        # default pose for LH (mirror of RH default)
        default = self._dexhand_default_dof_pos[None].repeat(n, 1)
        self._engine.set_dof_pos(env_ids, lh_id, default)
        self._engine.set_dof_vel(env_ids, lh_id, 0.0)

        # LH wrist: place at same z as RH but positive y offset
        import envs.dex_env as dex_env
        zpos = torch.full((n, 3), dex_env.TABLE_SURFACE_Z + dex_env.ROBOT_HEIGHT,
                          device=self._device, dtype=torch.float)
        zpos[:, 0] = -0.4
        zpos[:, 1] = 0.2
        self._engine.set_root_pos(env_ids, lh_id, zpos)

        lh_quat = torch.tensor([0.0, -0.70710678, 0.0, 0.70710678], device=self._device)
        lh_quat = lh_quat[None].repeat(n, 1)
        self._engine.set_root_rot(env_ids, lh_id, lh_quat)
        self._engine.set_root_vel(env_ids, lh_id, torch.zeros_like(zpos))
        self._engine.set_root_ang_vel(env_ids, lh_id, torch.zeros_like(zpos))
        return

    # ------------------------------------------------------------------ #
    # Reward — shared object + both hands                                #
    # ------------------------------------------------------------------ #
    def _update_reward(self):
        """PORT: dexhandmanip_bih.compute_reward.
        Bimanual reward = RH imitation + LH imitation + shared object tracking."""
        # reuse single-hand reward (imitation + obj trajectory)
        super()._update_reward()
        # TODO(port): add LH imitation term by mirroring the RH computation
        # using self._get_lh_dexhand_obj_id() and the LH demo data
        # (self._demo_data['lh_wrist_pos'] etc. – populate in _load_demo_data
        # when dexhandmanip_bih dataset is the source).
        return
