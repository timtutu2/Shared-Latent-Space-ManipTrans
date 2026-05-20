from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

import envs.base_env as base_env

try:
    import gym
    gym_spaces = gym.spaces
except Exception:
    import gymnasium as gym
    gym_spaces = gym.spaces

from rl_games.common import env_configurations, vecenv


def _to_gym_space(space):
    if isinstance(space, gym_spaces.Box):
        return gym_spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    if isinstance(space, gym_spaces.Dict):
        return gym_spaces.Dict({k: _to_gym_space(v) for k, v in space.spaces.items()})
    if isinstance(space, gym_spaces.Discrete):
        return gym_spaces.Discrete(space.n)

    # gymnasium object fallback by duck-typing
    cls_name = space.__class__.__name__.lower()
    if cls_name == "box":
        return gym_spaces.Box(low=np.array(space.low), high=np.array(space.high), shape=space.shape, dtype=space.dtype)
    if cls_name == "dict":
        return gym_spaces.Dict({k: _to_gym_space(v) for k, v in space.spaces.items()})
    if cls_name == "discrete":
        return gym_spaces.Discrete(int(space.n))

    raise TypeError(f"Unsupported space type: {type(space)}")


class ComplexObsRLGPUEnv(vecenv.IVecEnv):
    """rl_games vecenv adapter for ManipTrans_Lab envs (torch tensors, dict obs)."""

    def __init__(self, config_name):
        self.env = env_configurations.configurations[config_name]["env_creator"]()
        self._last_done = None

    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        obs, reward, done_flags, info = self.env.step(action)
        done = (done_flags != base_env.DoneFlags.NULL.value)
        self._last_done = done
        # Some rl_games play/test paths do not call reset_done() every frame.
        # To avoid "stuck-done" episodes, immediately reset terminated envs and
        # stitch reset observations back into the returned batch.
        done_ids = torch.nonzero(done, as_tuple=False).flatten()
        if done_ids.numel() > 0:
            reset_obs, _ = self.env.reset(done_ids)
            if isinstance(obs, dict):
                for k in obs.keys():
                    obs[k][done_ids] = reset_obs[k][done_ids]
            else:
                obs[done_ids] = reset_obs[done_ids]
        return obs, reward, done, info

    def reset(self):
        obs, _ = self.env.reset()
        self._last_done = torch.zeros(self.env.get_num_envs(), dtype=torch.bool, device=obs[next(iter(obs))].device)
        return obs

    def reset_done(self):
        if self._last_done is None:
            return self.reset()
        env_ids = torch.nonzero(self._last_done, as_tuple=False).flatten()
        if env_ids.numel() > 0:
            obs, _ = self.env.reset(env_ids)
            self._last_done[env_ids] = False
            return obs
        # no done envs: return current full obs buffer
        return self.env._obs_buf

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        obs_space = _to_gym_space(self.env.get_obs_space())
        action_space = _to_gym_space(self.env.get_action_space())
        info = {
            "action_space": action_space,
            "observation_space": obs_space,
        }
        if hasattr(self.env, "num_states") and self.env.num_states > 0 and hasattr(self.env, "state_space"):
            info["state_space"] = _to_gym_space(self.env.state_space)
        return info

    def set_train_info(self, env_frames, *args_, **kwargs_):
        if hasattr(self.env, "set_train_info"):
            self.env.set_train_info(env_frames, *args_, **kwargs_)

    def get_env_state(self):
        if hasattr(self.env, "get_env_state"):
            return self.env.get_env_state()
        return None

    def set_env_state(self, env_state):
        if hasattr(self.env, "set_env_state"):
            self.env.set_env_state(env_state)


def register_rlgpu_env(env_creator):
    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "RLGPU",
            "env_creator": env_creator,
        },
    )
    vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(config_name))
