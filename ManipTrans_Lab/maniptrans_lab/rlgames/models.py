from __future__ import annotations

from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

try:
    from gym import spaces
except Exception:
    from gymnasium import spaces

import rl_games.common.divergence as divergence
from rl_games.common.extensions.distributions import CategoricalMasked

from .moving_avg import RunningMeanStd, RunningMeanStdObs


class BaseModel:
    def __init__(self, model_class):
        self.model_class = model_class

    def build(self, config):
        obs_shape = config["input_shape"]
        normalize_value = config.get("normalize_value", False)
        normalize_input = config.get("normalize_input", False)
        normalize_input_excluded_keys = config.get("normalize_input_excluded_keys", None)
        value_size = config.get("value_size", 1)
        return self.Network(
            self.network_builder.build(self.model_class, **config),
            obs_shape=obs_shape,
            normalize_value=normalize_value,
            normalize_input=normalize_input,
            normalize_input_excluded_keys=normalize_input_excluded_keys,
            value_size=value_size,
            **(config.get("model", {})),
        )


class BaseModelNetwork(nn.Module):
    def __init__(self, obs_shape, normalize_value, normalize_input, value_size,
                 normalize_input_excluded_keys=None, **kwargs):
        super().__init__()
        self.obs_shape = obs_shape
        self.normalize_value = normalize_value
        self.normalize_input = normalize_input
        self.value_size = value_size

        if normalize_value:
            self.value_mean_std = RunningMeanStd((self.value_size,))
        if normalize_input:
            # rl_games may pass dict-observation shape as either:
            #   1) gym.spaces.Dict
            #   2) plain dict: {key: shape_tuple}
            if isinstance(obs_shape, spaces.Dict) or isinstance(obs_shape, dict):
                self.running_mean_std = RunningMeanStdObs(
                    obs_shape,
                    exclude_keys=normalize_input_excluded_keys,
                )
            else:
                self.running_mean_std = RunningMeanStd(obs_shape)

    def norm_obs(self, obs):
        with torch.no_grad():
            return self.running_mean_std(obs) if self.normalize_input else obs

    def denorm_value(self, value):
        with torch.no_grad():
            return self.value_mean_std(value, denorm=True) if self.normalize_value else value

    def get_aux_loss(self):
        return None


class ModelA2C(BaseModel):
    def __init__(self, network):
        super().__init__("a2c")
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            super().__init__(**kwargs)
            self.a2c_network = a2c_network

        def get_aux_loss(self):
            if hasattr(self.a2c_network, "get_aux_loss"):
                return self.a2c_network.get_aux_loss()
            return None

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def kl(self, p_dict, q_dict):
            return divergence.d_kl_discrete(p_dict["logits"], q_dict["logits"])

        def forward(self, input_dict):
            is_train = input_dict.get("is_train", True)
            action_masks = input_dict.get("action_masks", None)
            prev_actions = input_dict.get("prev_actions", None)
            input_dict["obs"] = self.norm_obs(input_dict["obs"])
            logits, value, states = self.a2c_network(input_dict)

            categorical = CategoricalMasked(logits=logits, masks=action_masks)
            if is_train:
                prev_neglogp = -categorical.log_prob(prev_actions)
                return {
                    "prev_neglogp": torch.squeeze(prev_neglogp),
                    "logits": categorical.logits,
                    "values": value,
                    "entropy": categorical.entropy(),
                    "rnn_states": states,
                }

            selected_action = categorical.sample().long()
            neglogp = -categorical.log_prob(selected_action)
            return {
                "neglogpacs": torch.squeeze(neglogp),
                "values": self.denorm_value(value),
                "actions": selected_action,
                "logits": categorical.logits,
                "rnn_states": states,
            }


class ModelA2CContinuousLogStd(BaseModel):
    def __init__(self, network):
        super().__init__("a2c")
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            super().__init__(**kwargs)
            self.a2c_network = a2c_network

        def get_aux_loss(self):
            if hasattr(self.a2c_network, "get_aux_loss"):
                return self.a2c_network.get_aux_loss()
            return None

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.get("is_train", True)
            prev_actions = input_dict.get("prev_actions", None)
            raw_obs = input_dict["obs"]
            input_dict["obs"] = self.norm_obs(raw_obs)
            # Residual networks may need unnormalized base observations for
            # frozen stage-1 policy normalization.
            input_dict["_raw_obs"] = raw_obs
            net_out = self.a2c_network(input_dict)
            base_actions = None
            if isinstance(net_out, (tuple, list)) and len(net_out) == 5:
                mu, logstd, value, states, base_actions = net_out
            else:
                mu, logstd, value, states = net_out

            # Numeric safety: prevent invalid std (NaN/Inf/<=0) from breaking sampling.
            mu = torch.nan_to_num(mu, nan=0.0, posinf=1e3, neginf=-1e3)
            logstd = torch.nan_to_num(logstd, nan=0.0, posinf=2.0, neginf=-20.0)
            logstd = torch.clamp(logstd, min=-20.0, max=2.0)
            sigma = torch.exp(logstd).clamp(min=1e-8, max=1e3)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                return {
                    "prev_neglogp": torch.squeeze(prev_neglogp),
                    "values": value,
                    "entropy": entropy,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                }
            selected_action = distr.sample()
            neglogp = self.neglogp(selected_action, mu, sigma, logstd)
            out = {
                "neglogpacs": torch.squeeze(neglogp),
                "values": self.denorm_value(value),
                "actions": selected_action,
                "rnn_states": states,
                "mus": mu,
                "sigmas": sigma,
            }
            if base_actions is not None:
                out["base_actions"] = base_actions
            return out

        @staticmethod
        def neglogp(x, mean, std, logstd):
            return (
                0.5 * (((x - mean) / std) ** 2).sum(dim=-1)
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1]
                + logstd.sum(dim=-1)
            )
