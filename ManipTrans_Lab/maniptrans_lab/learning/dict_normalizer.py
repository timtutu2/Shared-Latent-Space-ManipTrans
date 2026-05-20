"""A wrapper around one Normalizer per dict obs key."""
import gymnasium.spaces as spaces
import torch
import torch.nn as nn

import learning.normalizer as normalizer
import util.torch_util as torch_util


class DictNormalizer(nn.Module):
    def __init__(self, obs_space, device, clip=10.0):
        super().__init__()
        assert isinstance(obs_space, spaces.Dict)
        self._keys = sorted(list(obs_space.spaces.keys()))
        mods = {}
        for k in self._keys:
            sub = obs_space.spaces[k]
            dtype = torch_util.numpy_dtype_to_torch(sub.dtype)
            mods[k] = normalizer.Normalizer(sub.shape, clip=clip, device=device, dtype=dtype)
        self._norms = nn.ModuleDict(mods)

    def record(self, obs):
        for k in self._keys:
            if k in obs:
                self._norms[k].record(obs[k])

    def update(self):
        for k in self._keys:
            self._norms[k].update()

    def normalize(self, obs):
        if not isinstance(obs, dict):
            return obs
        return {k: self._norms[k].normalize(v) for k, v in obs.items() if k in self._norms}

    def get_mean(self):
        # return a flattened concatenation for logging compatibility
        return torch.cat([self._norms[k].get_mean().flatten() for k in self._keys])

    def get_std(self):
        return torch.cat([self._norms[k].get_std().flatten() for k in self._keys])
