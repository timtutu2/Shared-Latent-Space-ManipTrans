from __future__ import annotations

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    n = (name or "elu").lower()
    if n == "elu":
        return nn.ELU()
    if n in ("relu",):
        return nn.ReLU()
    if n in ("swish", "silu"):
        return nn.SiLU()
    if n in ("tanh",):
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class IdentityExtractor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.output_dim = int(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SimpleFeatureFusion(nn.Module):
    """ManipTrans-style dict feature fusion.

    Per-stream extractor (Identity by default) -> concat(sorted keys) -> MLP head.
    """

    def __init__(
        self,
        input_dims: dict[str, int],
        hidden_depth: int,
        hidden_dim: int,
        output_dim: int,
        activation: str,
        add_input_activation: bool,
        add_output_activation: bool,
    ):
        super().__init__()
        self._extractors = nn.ModuleDict(
            {k: IdentityExtractor(v) for k, v in input_dims.items()}
        )
        self._obs_groups = None
        self._obs_key_checked = False
        self.output_dim = int(output_dim)

        in_dim = sum(e.output_dim for e in self._extractors.values())
        layers: list[nn.Module] = []
        if add_input_activation:
            layers.append(_get_activation(activation))

        d = in_dim
        for _ in range(int(hidden_depth)):
            lin = nn.Linear(d, int(hidden_dim))
            nn.init.orthogonal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            layers.append(_get_activation(activation))
            d = int(hidden_dim)

        out = nn.Linear(d, self.output_dim)
        nn.init.orthogonal_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        if add_output_activation:
            layers.append(_get_activation(activation))

        self._head = nn.Sequential(*layers)

    def _group_obs(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        obs_keys = obs.keys()
        if self._obs_groups is None:
            obs_groups = {k.split("/")[0] for k in obs_keys}
            self._obs_groups = sorted(list(obs_groups))

        out = {}
        for g in self._obs_groups:
            is_subgroup = any(k.startswith(f"{g}/") for k in obs_keys)
            if is_subgroup:
                out[g] = {
                    k.split("/", 1)[1]: v for k, v in obs.items() if k.startswith(f"{g}/")
                }
            else:
                out[g] = obs[g]
        return out

    def _check_obs_key_match(self, obs: dict[str, torch.Tensor]):
        exp = set(self._extractors.keys())
        got = set(obs.keys())
        if exp != got:
            print(f"[rlgames] warning: obs key mismatch: expected={exp} got={got}")

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        obs = self._group_obs(obs)
        if not self._obs_key_checked:
            self._check_obs_key_match(obs)
            self._obs_key_checked = True

        feats = {}
        for k, ex in self._extractors.items():
            x = obs[k]
            if x.dim() > 2:
                x = x.flatten(start_dim=1)
            feats[k] = ex(x)

        x = torch.cat([feats[k] for k in sorted(feats.keys())], dim=-1)
        return self._head(x)


def build_from_config(input_dims: dict[str, int], cfg: dict) -> SimpleFeatureFusion:
    return SimpleFeatureFusion(
        input_dims=input_dims,
        hidden_depth=int(cfg.get("hidden_depth", 3)),
        hidden_dim=int(cfg.get("hidden_dim", 512)),
        output_dim=int(cfg.get("output_dim", 256)),
        activation=str(cfg.get("activation", "swish")),
        add_input_activation=bool(cfg.get("add_input_activation", False)),
        add_output_activation=bool(cfg.get("add_output_activation", False)),
    )
