from __future__ import annotations

from copy import deepcopy
import math
import torch
import torch.nn as nn

from rl_games.algos_torch.network_builder import A2CBuilder, NetworkBuilder

from .feature_encoder import build_from_config


class DictObsNetwork(A2CBuilder.Network):
    def __init__(self, params, **kwargs):
        actions_num = kwargs.pop("actions_num")
        self.value_size = kwargs.pop("value_size", 1)
        self.num_seqs = kwargs.pop("num_seqs", 1)

        NetworkBuilder.BaseNetwork.__init__(self)
        self.load(params)
        self.actor_cnn = nn.Sequential()
        self.critic_cnn = nn.Sequential()
        self.actor_mlp = nn.Sequential()
        self.critic_mlp = nn.Sequential()

        obs_shape = kwargs.get("input_shape")
        # rl_games passes Dict obs as:
        #   input_shape = {key: shape_tuple}
        # (not gym.spaces.Dict object). Support both forms.
        if isinstance(obs_shape, dict):
            input_dims = {}
            for k, shp in obs_shape.items():
                if isinstance(shp, (tuple, list)):
                    dim = int(shp[0])
                else:
                    dim = int(shp)
                input_dims[k] = dim
        elif hasattr(obs_shape, "spaces"):
            input_dims = {k: int(v.shape[0]) for k, v in obs_shape.spaces.items()}
        else:
            raise ValueError(
                f"dict_obs_actor_critic requires Dict obs input_shape, got: {type(obs_shape)}"
            )
        enc_cfg = dict(params.get("dict_feature_encoder", {}))
        self.dict_feature_encoder = build_from_config(input_dims, enc_cfg)
        mlp_input_shape = int(self.dict_feature_encoder.output_dim)

        in_mlp_shape = mlp_input_shape
        out_size = mlp_input_shape if len(self.units) == 0 else self.units[-1]

        if self.has_rnn:
            if not self.is_rnn_before_mlp:
                rnn_in_size = out_size
                out_size = self.rnn_units
                if self.rnn_concat_input:
                    rnn_in_size += in_mlp_shape
            else:
                rnn_in_size = in_mlp_shape
                in_mlp_shape = self.rnn_units

            if self.separate:
                self.a_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                self.c_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                if self.rnn_ln:
                    self.a_layer_norm = nn.LayerNorm(self.rnn_units)
                    self.c_layer_norm = nn.LayerNorm(self.rnn_units)
            else:
                self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                if self.rnn_ln:
                    self.layer_norm = nn.LayerNorm(self.rnn_units)

        mlp_args = {
            "input_size": in_mlp_shape,
            "units": self.units,
            "activation": self.activation,
            "norm_func_name": self.normalization,
            "dense_func": nn.Linear,
            "d2rl": self.is_d2rl,
            "norm_only_first_layer": self.norm_only_first_layer,
        }
        self.actor_mlp = self._build_mlp(**mlp_args)
        if self.separate:
            self.critic_mlp = self._build_mlp(**mlp_args)

        self.value = self._build_value_layer(out_size, self.value_size)
        self.value_act = self.activations_factory.create(self.value_activation)

        if self.is_discrete:
            self.logits = nn.Linear(out_size, actions_num)
        if self.is_multi_discrete:
            self.logits = nn.ModuleList([nn.Linear(out_size, num) for num in actions_num])
        if self.is_continuous:
            self.mu = nn.Linear(out_size, actions_num)
            self.mu_act = self.activations_factory.create(self.space_config["mu_activation"])
            mu_init = self.init_factory.create(**self.space_config["mu_init"])
            self.sigma_act = self.activations_factory.create(self.space_config["sigma_activation"])
            sigma_init = self.init_factory.create(**self.space_config["sigma_init"])

            if self.fixed_sigma:
                self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
                                          requires_grad=True)
            else:
                self.sigma = nn.Linear(out_size, actions_num)

        mlp_init = self.init_factory.create(**self.initializer)
        if self.has_cnn:
            cnn_init = self.init_factory.create(**self.cnn["initializer"])

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                cnn_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                mlp_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

        if self.is_continuous:
            mu_init(self.mu.weight)
            if self.fixed_sigma:
                sigma_init(self.sigma)
            else:
                sigma_init(self.sigma.weight)

    def forward(self, obs_dict):
        obs = obs_dict["obs"]
        states = obs_dict.get("rnn_states", None)
        dones = obs_dict.get("dones", None)
        bptt_len = obs_dict.get("bptt_len", 0)

        obs = self.dict_feature_encoder(obs)

        if self.separate:
            a_out = c_out = obs
            a_out = self.actor_cnn(a_out).contiguous().view(a_out.size(0), -1)
            c_out = self.critic_cnn(c_out).contiguous().view(c_out.size(0), -1)
            if self.has_rnn:
                seq_length = obs_dict.get("seq_length", 1)
                if not self.is_rnn_before_mlp:
                    a_out_in, c_out_in = a_out, c_out
                    a_out = self.actor_mlp(a_out_in)
                    c_out = self.critic_mlp(c_out_in)
                    if self.rnn_concat_input:
                        a_out = torch.cat([a_out, a_out_in], dim=1)
                        c_out = torch.cat([c_out, c_out_in], dim=1)
                batch_size = a_out.size()[0]
                num_seqs = batch_size // seq_length
                a_out = a_out.reshape(num_seqs, seq_length, -1).transpose(0, 1)
                c_out = c_out.reshape(num_seqs, seq_length, -1).transpose(0, 1)
                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1).transpose(0, 1)
                if len(states) == 2:
                    a_states, c_states = states[0], states[1]
                else:
                    a_states, c_states = states[:2], states[2:]
                a_out, a_states = self.a_rnn(a_out, a_states, dones, bptt_len)
                c_out, c_states = self.c_rnn(c_out, c_states, dones, bptt_len)
                a_out = a_out.transpose(0, 1).contiguous().reshape(-1, a_out.size(-1))
                c_out = c_out.transpose(0, 1).contiguous().reshape(-1, c_out.size(-1))
                if self.rnn_ln:
                    a_out = self.a_layer_norm(a_out)
                    c_out = self.c_layer_norm(c_out)
                if type(a_states) is not tuple:
                    a_states = (a_states,)
                    c_states = (c_states,)
                states = a_states + c_states
                if self.is_rnn_before_mlp:
                    a_out = self.actor_mlp(a_out)
                    c_out = self.critic_mlp(c_out)
            else:
                a_out = self.actor_mlp(a_out)
                c_out = self.critic_mlp(c_out)

            value = self.value_act(self.value(c_out))
            if self.is_discrete:
                return self.logits(a_out), value, states
            if self.is_multi_discrete:
                return [l(a_out) for l in self.logits], value, states
            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                sigma = (mu * 0.0 + self.sigma_act(self.sigma)) if self.fixed_sigma else self.sigma_act(self.sigma(a_out))
                return mu, sigma, value, states

        out = self.actor_cnn(obs).flatten(1)
        if self.has_rnn:
            seq_length = obs_dict.get("seq_length", 1)
            out_in = out
            if not self.is_rnn_before_mlp:
                out = self.actor_mlp(out)
                if self.rnn_concat_input:
                    out = torch.cat([out, out_in], dim=1)
            batch_size = out.size()[0]
            num_seqs = batch_size // seq_length
            out = out.reshape(num_seqs, seq_length, -1)
            if len(states) == 1:
                states = states[0]
            out = out.transpose(0, 1)
            if dones is not None:
                dones = dones.reshape(num_seqs, seq_length, -1).transpose(0, 1)
            out, states = self.rnn(out, states, dones, bptt_len)
            out = out.transpose(0, 1).contiguous().reshape(-1, out.size(-1))
            if self.rnn_ln:
                out = self.layer_norm(out)
            if self.is_rnn_before_mlp:
                out = self.actor_mlp(out)
            if type(states) is not tuple:
                states = (states,)
        else:
            out = self.actor_mlp(out)

        value = self.value_act(self.value(out))
        if self.central_value:
            return value, states
        if self.is_discrete:
            return self.logits(out), value, states
        if self.is_multi_discrete:
            return [l(out) for l in self.logits], value, states
        if self.is_continuous:
            mu = self.mu_act(self.mu(out))
            sigma = self.sigma_act(self.sigma) if self.fixed_sigma else self.sigma_act(self.sigma(out))
            return mu, mu * 0 + sigma, value, states

    def re_initialize_mu_gripper(self):
        nn.init.kaiming_uniform_(self.mu.weight[-1:, :], a=math.sqrt(5))
        if self.mu.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.mu.weight[-1:, :])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.mu.bias[-1:], -bound, bound)


class DictObsBuilder(A2CBuilder):
    def build(self, name, **kwargs):
        return DictObsNetwork(self.params, **kwargs)


class ResDictObsNetwork(DictObsNetwork):
    """Residual single-hand network for rl_games:
    action = [frozen_stage1_base_action, trainable_residual_action].
    """

    def __init__(self, params, **kwargs):
        res_actions_num = int(kwargs.get("actions_num"))
        self._use_pid_control = bool(params.get("use_pid_control", False))
        self._use_quat_rot = bool(params.get("use_quat_rot", False))
        self._n_dof = self._infer_n_dof(res_actions_num, self._use_quat_rot)
        self._base_action_dim = (9 + self._n_dof) if self._use_pid_control else (6 + self._n_dof)
        self._res_action_dim = res_actions_num

        # Build residual branch with residual-action output size.
        kwargs = dict(kwargs)
        kwargs["actions_num"] = self._res_action_dim
        super().__init__(params, **kwargs)
        # Match ManipTrans residual network behavior: residual actor/value branch
        # is conditioned on sampled stage-1 base action.
        self._rebuild_residual_mlps()

        self._residual_base_ckpt = str(params.get("residual_base_model_checkpoint", "") or "")
        if self._residual_base_ckpt == "":
            raise ValueError(
                "res_dict_obs_actor_critic requires network.residual_base_model_checkpoint in rlg_config/CLI."
            )

        self._base_obs_shape = self._build_base_obs_shape(kwargs.get("input_shape"))
        self._base_obs_shape = self._align_base_obs_shape_to_checkpoint(
            self._base_obs_shape, self._residual_base_ckpt
        )
        base_kwargs = dict(kwargs)
        base_kwargs["actions_num"] = self._base_action_dim
        base_kwargs["input_shape"] = self._base_obs_shape
        base_params = deepcopy(params)
        base_params.pop("residual_base_model_checkpoint", None)
        self._base_model = DictObsNetwork(base_params, **base_kwargs)
        self._load_base_checkpoint(self._residual_base_ckpt)
        for p in self._base_model.parameters():
            p.requires_grad = False
        self._base_model.eval()

        self._base_obs_norm = self._extract_base_obs_norm(self._residual_base_ckpt)

    def _rebuild_residual_mlps(self):
        # DictObsNetwork builds actor_mlp assuming encoder-only input.
        # Residual policy needs [encoder_features, base_action] input.
        in_mlp_shape = int(self.dict_feature_encoder.output_dim) + int(self._base_action_dim)
        mlp_args = {
            "input_size": in_mlp_shape,
            "units": self.units,
            "activation": self.activation,
            "norm_func_name": self.normalization,
            "dense_func": nn.Linear,
            "d2rl": self.is_d2rl,
            "norm_only_first_layer": self.norm_only_first_layer,
        }
        self.actor_mlp = self._build_mlp(**mlp_args)
        if self.separate:
            self.critic_mlp = self._build_mlp(**mlp_args)

    @staticmethod
    def _infer_n_dof(res_actions_num: int, use_quat_rot: bool) -> int:
        root_dim = 7 if use_quat_rot else 6
        n = res_actions_num - root_dim
        if n <= 0:
            raise ValueError(
                f"Cannot infer dof count from residual actions={res_actions_num}, use_quat_rot={use_quat_rot}"
            )
        return int(n)

    def _build_base_obs_shape(self, obs_shape):
        if not isinstance(obs_shape, dict):
            return obs_shape
        out = {}
        for k, shp in obs_shape.items():
            dim = int(shp[0] if isinstance(shp, (tuple, list)) else shp)
            if k == "privileged":
                dim = min(dim, self._n_dof)
            out[k] = (dim,)
        return out

    def _align_base_obs_shape_to_checkpoint(self, base_obs_shape, ckpt_path):
        """Match frozen base branch input dims to stage-1 checkpoint stats.

        This is required when stage-2 augments obs (e.g., +BPS in target):
        residual branch should see full obs, while frozen stage-1 branch must
        keep its original input widths.
        """
        if not isinstance(base_obs_shape, dict):
            return base_obs_shape
        try:
            ckpt = self._safe_torch_load(ckpt_path)
            model_sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        except Exception:
            return base_obs_shape

        out = dict(base_obs_shape)
        for k in ("proprioception", "privileged", "target"):
            mean_k = f"running_mean_std.running_mean_std.{k}.running_mean"
            if mean_k in model_sd:
                ckpt_dim = int(model_sd[mean_k].numel())
                if k in out:
                    cur_dim = int(out[k][0] if isinstance(out[k], (tuple, list)) else out[k])
                    out[k] = (min(cur_dim, ckpt_dim),)
                else:
                    out[k] = (ckpt_dim,)
        return out

    @staticmethod
    def _safe_torch_load(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _load_base_checkpoint(self, path):
        ckpt = self._safe_torch_load(path)
        model_sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        a2c_sd = {
            k.replace("a2c_network.", "", 1): v
            for k, v in model_sd.items()
            if k.startswith("a2c_network.")
        }
        missing, unexpected = self._base_model.load_state_dict(a2c_sd, strict=False)
        if len(missing) > 0:
            print(f"[res_dict_obs_actor_critic] base model missing keys: {len(missing)}")
        if len(unexpected) > 0:
            print(f"[res_dict_obs_actor_critic] base model unexpected keys: {len(unexpected)}")

    def _extract_base_obs_norm(self, path):
        ckpt = self._safe_torch_load(path)
        model_sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        out = {}
        for k in ("proprioception", "privileged", "target"):
            mean_k = f"running_mean_std.running_mean_std.{k}.running_mean"
            var_k = f"running_mean_std.running_mean_std.{k}.running_var"
            if mean_k in model_sd and var_k in model_sd:
                mean = model_sd[mean_k].detach().clone()
                var = model_sd[var_k].detach().clone()
                std = torch.sqrt(torch.clamp(var, min=1e-8))
                out[k] = {"mean": mean, "std": std}
        return out

    def _slice_base_obs(self, obs):
        if not isinstance(obs, dict):
            return obs
        out = {}
        for k, v in obs.items():
            if k in self._base_obs_shape:
                dim = int(self._base_obs_shape[k][0])
                out[k] = v[..., :dim]
        return out

    def _normalize_base_obs(self, obs):
        if not isinstance(obs, dict):
            return obs
        out = {}
        for k, v in obs.items():
            if k in self._base_obs_norm:
                mean = self._base_obs_norm[k]["mean"].to(device=v.device, dtype=v.dtype)
                std = self._base_obs_norm[k]["std"].to(device=v.device, dtype=v.dtype)
                out[k] = torch.clamp((v - mean) / torch.clamp(std, min=1e-5), -10.0, 10.0)
            else:
                out[k] = v
        return out

    def forward(self, obs_dict):
        if self.separate:
            raise NotImplementedError("res_dict_obs_actor_critic does not support separate=true")
        if self.has_rnn:
            raise NotImplementedError("res_dict_obs_actor_critic does not support RNN")

        # Base branch (frozen stage-1)
        with torch.no_grad():
            raw_obs = obs_dict.get("_raw_obs", obs_dict["obs"])
            base_obs = self._slice_base_obs(raw_obs)
            base_obs = self._normalize_base_obs(base_obs)
            base_in = {"obs": base_obs}
            base_mu, base_logstd, _, _ = self._base_model(base_in)
            base_logstd = torch.nan_to_num(base_logstd, nan=0.0, posinf=2.0, neginf=-20.0)
            base_logstd = torch.clamp(base_logstd, min=-20.0, max=2.0)
            base_sigma = torch.exp(base_logstd).clamp(min=1e-8, max=1e3)
            base_distr = torch.distributions.Normal(base_mu, base_sigma, validate_args=False)
            base_action = base_distr.sample()

        # Residual branch (trainable), conditioned on sampled base action.
        states = obs_dict.get("rnn_states", None)
        obs = self.dict_feature_encoder(obs_dict["obs"])
        out = self.actor_cnn(obs).flatten(1)
        out = torch.cat([out, base_action.detach()], dim=-1)
        out = self.actor_mlp(out)

        value = self.value_act(self.value(out))
        res_mu = self.mu_act(self.mu(out))
        if self.fixed_sigma:
            res_logstd = res_mu * 0 + self.sigma_act(self.sigma)
        else:
            res_logstd = self.sigma_act(self.sigma(out))

        # Return residual distribution only; rollout code will concatenate
        # [base_action, residual_action] when stepping the env.
        return res_mu, res_logstd, value, states, base_action.detach()


class ResDictObsBuilder(A2CBuilder):
    def build(self, name, **kwargs):
        return ResDictObsNetwork(self.params, **kwargs)
