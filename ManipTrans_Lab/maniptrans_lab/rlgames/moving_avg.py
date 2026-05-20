from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    from gym import spaces
except Exception:
    from gymnasium import spaces

from rl_games.algos_torch import torch_ext


class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-5, per_channel=False, norm_only=False):
        super().__init__()
        self.insize = insize
        self.epsilon = epsilon
        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0, 2, 3]
            elif len(self.insize) == 2:
                self.axis = [0, 2]
            elif len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0]
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    @staticmethod
    def _update(mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = m2 / tot_count
        return new_mean, new_var, tot_count

    def forward(self, x, denorm=False, mask=None):
        if self.training:
            if mask is not None:
                mean, var = torch_ext.get_mean_std_with_masks(x, mask)
            else:
                mean = x.mean(self.axis)
                var = x.var(self.axis)
            self.running_mean, self.running_var, self.count = self._update(
                self.running_mean, self.running_var, self.count, mean, var, x.size()[0]
            )

        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view([1, self.insize[0], 1, 1]).expand_as(x)
                current_var = self.running_var.view([1, self.insize[0], 1, 1]).expand_as(x)
            elif len(self.insize) == 2:
                current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(x)
                current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(x)
            else:
                current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(x)
                current_var = self.running_var.view([1, self.insize[0]]).expand_as(x)
        else:
            current_mean = self.running_mean
            current_var = self.running_var

        if denorm:
            y = torch.clamp(x, min=-20.0, max=20.0)
            y = torch.sqrt(current_var.float() + self.epsilon) * y + current_mean.float()
        else:
            if self.norm_only:
                y = x / torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (x - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=-20.0, max=20.0)
        return y


class RunningMeanStdObs(nn.Module):
    def __init__(self, insize, epsilon=1e-5, per_channel=False, norm_only=False, exclude_keys=None):
        super().__init__()
        exclude_keys = exclude_keys or []
        self._exclude_keys = list(exclude_keys)
        # Support both gym.spaces.Dict and rl_games shape dict ({k: tuple/int}).
        if isinstance(insize, spaces.Dict):
            items = {k: v.shape for k, v in insize.items()}
        elif isinstance(insize, dict):
            items = {}
            for k, v in insize.items():
                if isinstance(v, (tuple, list)):
                    items[k] = tuple(v)
                else:
                    items[k] = (int(v),)
        else:
            raise TypeError(f"RunningMeanStdObs expects Dict space or shape dict, got {type(insize)}")

        self.running_mean_std = nn.ModuleDict(
            {k: RunningMeanStd(shape, epsilon, per_channel, norm_only) for k, shape in items.items()
             if k not in self._exclude_keys}
        )

    def forward(self, x, denorm=False):
        return {
            k: (self.running_mean_std[k](v, denorm) if k not in self._exclude_keys else v)
            for k, v in x.items()
        }
