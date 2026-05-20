"""
Dict-obs variant of MimicKit's PPOModel. Used for ManipTrans imitator / manipulation
tasks whose observation is a gymnasium.spaces.Dict
({"proprioception", "privileged", "target"}).

Pipeline: obs dict -> DictFeatureFusion net -> shared MLP trunk -> action dist / value head.
The actor and critic can optionally use different obs subsets (e.g. the actor only
sees proprioception+target while the critic sees all three, matching ManipTrans's
asymmetric actor-critic setup).
"""
import gymnasium.spaces as spaces
import numpy as np
import torch
import torch.nn as nn

import learning.base_model as base_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util


class DictPPOModel(base_model.BaseModel):
    def __init__(self, config, env):
        super().__init__(config, env)

        self._actor_obs_keys = config.get("actor_obs_keys", ["proprioception", "target"])
        self._critic_obs_keys = config.get("critic_obs_keys",
                                           ["proprioception", "privileged", "target"])
        # Stage-2 residual mode: condition actor on frozen stage-1 base policy action.
        self._residual_base_model_file = config.get("residual_base_model_file", "")
        self._residual_base_action_mode = config.get("residual_base_action_mode", "sample")
        self._residual_mlp_units = config.get("residual_mlp_units", [512, 128, 64])
        self._residual_base_action_dim_cfg = config.get("residual_base_action_dim", None)
        self._residual_enabled = self._residual_base_model_file not in ("", None)
        self._base_model = None
        self._base_obs_norm = None
        self._base_action_dim = self._infer_base_action_dim(env)

        self._build_nets(config, env)
        if self._residual_enabled:
            self._build_residual_base_model(config, env)
        return

    def _infer_base_action_dim(self, env):
        if isinstance(self._residual_base_action_dim_cfg, int) and self._residual_base_action_dim_cfg > 0:
            return int(self._residual_base_action_dim_cfg)
        if not self._residual_enabled:
            if isinstance(env.get_action_space(), spaces.Box):
                return int(np.prod(env.get_action_space().shape))
            return 0
        # Stage-2 residual tasks (DexManipSH) use action = [base_action, residual_action].
        # The frozen stage-1 policy should output only base_action.
        if hasattr(env, "_dexhand") and hasattr(env, "_use_pid_control"):
            n_dof = int(env._dexhand.n_dofs)
            root_ctrl_dim = 9 if bool(env._use_pid_control) else 6
            return root_ctrl_dim + n_dof
        if isinstance(env.get_action_space(), spaces.Box):
            return int(np.prod(env.get_action_space().shape))
        return 0

    def eval_actor(self, obs):
        actor_in = self._select_obs(obs, self._actor_obs_keys)
        h = self._actor_layers(actor_in)
        if self._residual_enabled:
            base_action = self._eval_base_action(obs)
            h = self._residual_actor_mlp(torch.cat([h, base_action], dim=-1))
        return self._action_dist(h)

    def eval_critic(self, obs):
        critic_in = self._select_obs(obs, self._critic_obs_keys)
        h = self._critic_layers(critic_in)
        return self._critic_out(h)

    def get_actor_params(self):
        params = list(self._actor_layers.parameters())
        if self._residual_enabled:
            params += list(self._residual_actor_mlp.parameters())
        params += list(self._action_dist.parameters())
        return params

    def get_critic_params(self):
        return list(self._critic_layers.parameters()) + list(self._critic_out.parameters())

    def _select_obs(self, obs, keys):
        if isinstance(obs, dict):
            return {k: obs[k] for k in keys if k in obs}
        return obs

    def _build_nets(self, config, env):
        self._build_actor(config, env)
        self._build_critic(config, env)
        return

    def _build_actor(self, config, env):
        net_name = config["actor_net"]
        input_dict = self._build_input_dict(env, self._actor_obs_keys)
        self._actor_layers, _ = net_builder.build_net(net_name, input_dict,
                                                     activation=self._activation)
        if self._residual_enabled:
            in_size = torch_util.calc_layers_out_size(self._actor_layers) + self._base_action_dim
            layers = []
            for out_size in self._residual_mlp_units:
                lin = nn.Linear(in_size, out_size)
                nn.init.zeros_(lin.bias)
                layers.append(lin)
                layers.append(self._activation())
                in_size = out_size
            self._residual_actor_mlp = nn.Sequential(*layers)
            self._action_dist = self._build_action_distribution(config, env, self._residual_actor_mlp)
        else:
            self._action_dist = self._build_action_distribution(config, env, self._actor_layers)
        return

    def _build_critic(self, config, env):
        net_name = config["critic_net"]
        input_dict = self._build_input_dict(env, self._critic_obs_keys)
        self._critic_layers, _ = net_builder.build_net(net_name, input_dict,
                                                      activation=self._activation)
        out_size = torch_util.calc_layers_out_size(self._critic_layers)
        self._critic_out = torch.nn.Linear(out_size, 1)
        torch.nn.init.zeros_(self._critic_out.bias)
        return

    def _build_input_dict(self, env, keys):
        obs_space = env.get_obs_space()
        if isinstance(obs_space, spaces.Dict):
            return {k: obs_space.spaces[k] for k in keys if k in obs_space.spaces}
        return {"obs": obs_space}

    def _build_residual_base_model(self, config, env):
        class _BaseEnvProxy:
            def __init__(self, obs_space, action_dim):
                self._obs_space = obs_space
                self._act_space = spaces.Box(
                    low=-np.ones(action_dim, dtype=np.float32),
                    high=np.ones(action_dim, dtype=np.float32),
                    dtype=np.float32,
                )

            def get_obs_space(self):
                return self._obs_space

            def get_action_space(self):
                return self._act_space

        base_cfg = dict(config)
        # Prevent recursive residual loading.
        base_cfg["residual_base_model_file"] = ""
        base_env = _BaseEnvProxy(env.get_obs_space(), self._base_action_dim)
        self._base_model = DictPPOModel(base_cfg, base_env)

        ckpt = torch.load(self._residual_base_model_file, map_location="cpu")
        model_sd = {
            k.replace("_model.", "", 1): v
            for k, v in ckpt.items()
            if k.startswith("_model.")
        }
        missing, unexpected = self._base_model.load_state_dict(model_sd, strict=False)
        if len(missing) > 0:
            print(f"[DictPPOModel] base model missing keys: {len(missing)}")
        if len(unexpected) > 0:
            print(f"[DictPPOModel] base model unexpected keys: {len(unexpected)}")

        # Load stage-1 obs normalizer stats so base policy sees same normalization.
        norm = {}
        for key in ("proprioception", "privileged", "target"):
            mean_k = f"_obs_norm._norms.{key}._mean"
            std_k = f"_obs_norm._norms.{key}._std"
            if mean_k in ckpt and std_k in ckpt:
                norm[key] = {
                    "mean": ckpt[mean_k].detach().clone(),
                    "std": torch.clamp(ckpt[std_k].detach().clone(), min=1e-5),
                }
        self._base_obs_norm = norm

        for p in self._base_model.parameters():
            p.requires_grad = False
        self._base_model.eval()
        return

    def _normalize_for_base(self, obs):
        out = {}
        for k, v in obs.items():
            if k in self._base_obs_norm:
                mean = self._base_obs_norm[k].get("mean")
                std = self._base_obs_norm[k].get("std")
                if mean is not None and std is not None:
                    mean = mean.to(v.device)
                    std = std.to(v.device)
                    out[k] = torch.clamp((v - mean) / std, -10.0, 10.0)
                    continue
            out[k] = v
        return out

    def _eval_base_action(self, obs):
        with torch.no_grad():
            base_obs = self._select_obs(obs, self._actor_obs_keys)
            base_obs = self._normalize_for_base(base_obs)
            dist = self._base_model.eval_actor(base_obs)
            if self._residual_base_action_mode == "mode":
                a = dist.mode
            else:
                a = dist.sample()
        return a.detach()
