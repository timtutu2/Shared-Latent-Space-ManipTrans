"""Dict-obs feature fusion net for ManipTrans-style {proprioception, privileged, target}
observations.

Registered with net_builder exactly like MimicKit's fc_2layers_*.py modules.
Mirrors the logic of ManipTrans/lib/nn/features/fusion.py's SimpleFeatureFusion:
each obs stream runs through its own identity/Linear extractor, features are
concatenated (sorted by key for determinism), then passed through an MLP head.
"""
import numpy as np
import torch
import torch.nn as nn


class DictFeatureFusion(nn.Module):
    def __init__(self, input_dict, hidden_sizes, activation):
        super().__init__()
        self._keys = sorted(list(input_dict.keys()))
        self._extractors = nn.ModuleDict()
        total_in = 0
        for k in self._keys:
            in_dim = int(np.prod(input_dict[k].shape))
            # identity extractor (matches ManipTrans default: Identity for proprio/priv/target)
            self._extractors[k] = nn.Identity()
            total_in += in_dim

        layers = []
        in_size = total_in
        for out_size in hidden_sizes:
            lin = nn.Linear(in_size, out_size)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            layers.append(activation())
            in_size = out_size

        self._head = nn.Sequential(*layers)
        self._out_size = in_size

    def forward(self, obs):
        if not isinstance(obs, dict):
            # treat flat tensor as a single 'obs' stream
            return self._head(obs)
        feats = []
        for k in self._keys:
            if k not in obs:
                continue
            x = obs[k]
            x = self._extractors[k](x.flatten(start_dim=1))
            feats.append(x)
        x = torch.cat(feats, dim=-1)
        return self._head(x)


def build_net(input_dict, activation):
    hidden_sizes = [1024, 512]
    net = DictFeatureFusion(input_dict, hidden_sizes=hidden_sizes, activation=activation)
    info = dict()
    return net, info
