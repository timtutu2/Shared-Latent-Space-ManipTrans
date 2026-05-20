"""Run ManipTrans_Lab environments with rl_games PPO.

This is a compatibility path to stay close to original ManipTrans (Isaac Gym + rl_games)
while still using this Isaac-Lab environment port.
"""
from __future__ import annotations

import copy
import math
import os
import sys
import time
import yaml
import torch

import numpy as np

import envs.base_env as base_env
import envs.env_builder as env_builder
import util.arg_parser as arg_parser
import util.util as util

from util.logger import Logger


def _load_args(argv):
    args = arg_parser.ArgParser()
    args.load_args(argv[1:])
    arg_file = args.parse_string("arg_file")
    if arg_file != "":
        succ = args.load_file(arg_file)
        assert succ, Logger.print(f"Failed to load args from: {arg_file}")
    return args


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _set_seed(seed: int):
    print(f"[rlgames] Setting seed: {seed}")
    util.set_rand_seed(seed)


def _import_rlgames_or_die():
    try:
        from rl_games.algos_torch.model_builder import register_model, register_network
        from rl_games.torch_runner import Runner
        return register_model, register_network, Runner
    except Exception as e:
        raise RuntimeError(
            "rl_games is not installed. Install with: `pip install rl-games` "
            "(and ensure gym/gymnasium deps are present)."
        ) from e


def main(argv):
    args = _load_args(argv)

    mode = args.parse_string("mode", "train")
    assert mode in ("train", "test"), f"Unsupported mode: {mode}"

    num_envs = args.parse_int("num_envs", 4096)
    visualize = args.parse_bool("visualize", mode == "test")
    devices = args.parse_strings("devices", ["cuda:0"])
    device = devices[0]

    env_file = args.parse_string("env_config")
    engine_file = args.parse_string("engine_config")

    rlg_cfg_file = args.parse_string(
        "rlg_config",
        "data/rl_games/dex_imitator_ppo_rlgames.yaml",
    )
    out_dir = args.parse_string("out_dir", "output/dex_imitator_allegro_rh_ppo_rlgames")
    model_file = args.parse_string("model_file", "")
    residual_base_ckpt = args.parse_string("residual_base_model_checkpoint", "")

    seed = args.parse_int("rand_seed", int(time.time()) % (2**31 - 1))
    _set_seed(seed)

    register_model, register_network, Runner = _import_rlgames_or_die()

    from rlgames.models import ModelA2CContinuousLogStd
    from rlgames.network_builder import DictObsBuilder, ResDictObsBuilder
    from rlgames.env_wrapper import register_rlgpu_env

    register_model("my_continuous_a2c_logstd", ModelA2CContinuousLogStd)
    register_network("dict_obs_actor_critic", DictObsBuilder)
    register_network("res_dict_obs_actor_critic", ResDictObsBuilder)

    def create_env():
        env = env_builder.build_env(
            env_file=env_file,
            engine_file=engine_file,
            num_envs=num_envs,
            device=device,
            visualize=visualize,
            record_video=False,
        )
        env.set_mode(base_env.EnvMode.TEST if mode == "test" else base_env.EnvMode.TRAIN)
        return env

    register_rlgpu_env(create_env)

    cfg = _load_yaml(rlg_cfg_file)
    cfg = copy.deepcopy(cfg)
    env_cfg = _load_yaml(env_file)

    # Overwrite runtime-critical fields from CLI.
    params = cfg.setdefault("params", {})
    params["seed"] = seed
    cfg_net = params.setdefault("network", {})
    cfg_model = params.setdefault("model", {})
    cfg_algo = params.setdefault("algo", {})
    cfg_conf = params.setdefault("config", {})

    # rl_games default runner key for PPO-style continuous control.
    cfg_algo["name"] = "a2c_continuous"
    cfg_model["name"] = "my_continuous_a2c_logstd"
    cfg_net["name"] = "dict_obs_actor_critic"
    if residual_base_ckpt != "":
        cfg_net["name"] = "res_dict_obs_actor_critic"
        cfg_net["residual_base_model_checkpoint"] = residual_base_ckpt
        cfg_net["use_pid_control"] = bool(env_cfg.get("usePIDControl", False))
        cfg_net["use_quat_rot"] = bool(env_cfg.get("useQuatRot", False))
    cfg_conf["env_name"] = "rlgpu"
    cfg_conf["num_actors"] = num_envs
    # Keep rl_games model device aligned with env tensors.
    cfg_conf["device"] = device
    cfg_conf["train_dir"] = out_dir
    # For play/test path, force vecenv so rl_games doesn't expect gym-style
    # observation_space/action_space attrs on raw DexImitatorEnv.
    player_cfg = cfg_conf.setdefault("player", {})
    player_cfg["use_vecenv"] = True
    if mode == "test":
        # rl_games play mode may exit after a very small number of episodes
        # unless games_num is explicitly set.
        player_cfg["games_num"] = args.parse_int("games_num", 1000)
        player_cfg["print_stats"] = args.parse_bool("print_stats", True)

    # keep naming close to old ManipTrans runs
    exp_name = args.parse_string("experiment", "")
    if exp_name != "":
        cfg_conf["name"] = exp_name
    cfg_conf["full_experiment_name"] = cfg_conf.get("name", "dex_imitator_rlgames")

    if args.has_key("learning_rate"):
        cfg_conf["learning_rate"] = args.parse_float("learning_rate")
    if args.has_key("max_iterations"):
        cfg_conf["max_epochs"] = args.parse_int("max_iterations")
    if args.has_key("save_frequency"):
        cfg_conf["save_frequency"] = args.parse_int("save_frequency")
    if args.has_key("minibatch_size"):
        cfg_conf["minibatch_size"] = args.parse_int("minibatch_size")
    if args.has_key("save_best_after"):
        cfg_conf["save_best_after"] = args.parse_int("save_best_after")
    if args.has_key("score_to_win"):
        cfg_conf["score_to_win"] = args.parse_float("score_to_win")
    if args.has_key("print_stats"):
        cfg_conf["print_stats"] = args.parse_bool("print_stats")

    # rl_games requires batch_size % minibatch_size == 0
    # batch_size = num_actors * horizon_length.
    horizon = int(cfg_conf.get("horizon_length", 32))
    batch_size = int(num_envs) * horizon
    minibatch = int(cfg_conf.get("minibatch_size", batch_size))
    if minibatch <= 0:
        minibatch = batch_size
    if batch_size % minibatch != 0:
        # Pick the largest valid divisor <= requested minibatch.
        fixed = math.gcd(batch_size, minibatch)
        if fixed <= 0:
            fixed = batch_size
        print(
            f"[rlgames] adjust minibatch_size: requested={minibatch}, "
            f"batch_size={batch_size} -> {fixed}"
        )
        cfg_conf["minibatch_size"] = fixed

    os.makedirs(out_dir, exist_ok=True)
    dumped_cfg = os.path.join(out_dir, "rlgames_config_resolved.yaml")
    with open(dumped_cfg, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[rlgames] resolved config: {dumped_cfg}")

    # Match original ManipTrans residual rollout path:
    # model samples residual action and also returns frozen base action;
    # env step receives concatenated [base_action, residual_action].
    if residual_base_ckpt != "":
        from rl_games.common import a2c_common
        from rl_games.algos_torch import players as rlg_players
        if not getattr(a2c_common.A2CBase, "_maniptrans_residual_concat_patched", False):
            _orig_get_action_values = a2c_common.A2CBase.get_action_values
            _orig_env_step = a2c_common.A2CBase.env_step

            def _patched_get_action_values(self, obs):
                res = _orig_get_action_values(self, obs)
                self._mtlab_base_actions = res.get("base_actions", None)
                return res

            def _patched_env_step(self, actions):
                base_actions = getattr(self, "_mtlab_base_actions", None)
                self._mtlab_base_actions = None
                if not isinstance(base_actions, torch.Tensor):
                    return _orig_env_step(self, actions)

                # 1) preprocess residual actions using residual action-space bounds
                # 2) concatenate frozen base actions
                # 3) step env with full action [base, residual]
                proc_actions = self.preprocess_actions(actions)
                if not isinstance(proc_actions, torch.Tensor):
                    proc_actions = torch.as_tensor(proc_actions, device=self.ppo_device)
                base_actions = base_actions.to(proc_actions.device)
                full_actions = torch.cat([torch.clamp(base_actions, -1.0, 1.0), proc_actions], dim=1)

                obs, rewards, dones, infos = self.vec_env.step(full_actions)
                if self.is_tensor_obses:
                    if self.value_size == 1:
                        rewards = rewards.unsqueeze(1)
                    return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
                else:
                    if self.value_size == 1:
                        rewards = np.expand_dims(rewards, axis=1)
                    return (
                        self.obs_to_tensors(obs),
                        torch.from_numpy(rewards).to(self.ppo_device).float(),
                        torch.from_numpy(dones).to(self.ppo_device),
                        infos,
                    )

            a2c_common.A2CBase.get_action_values = _patched_get_action_values
            a2c_common.A2CBase.env_step = _patched_env_step
            a2c_common.A2CBase._maniptrans_residual_concat_patched = True

        # Match residual concat in play/test path too.
        if not getattr(rlg_players.PpoPlayerContinuous, "_maniptrans_residual_concat_patched", False):
            def _patched_player_get_action(self, obs, is_deterministic=False):
                if self.has_batch_dimension == False:
                    obs = rlg_players.unsqueeze_obs(obs)
                obs = self._preproc_obs(obs)
                input_dict = {
                    "is_train": False,
                    "prev_actions": None,
                    "obs": obs,
                    "rnn_states": self.states,
                }
                with torch.no_grad():
                    res_dict = self.model(input_dict)
                mu = res_dict["mus"]
                action = res_dict["actions"]
                self.states = res_dict["rnn_states"]
                current_action = mu if is_deterministic else action
                base_actions = res_dict.get("base_actions", None)

                if self.has_batch_dimension == False:
                    current_action = torch.squeeze(current_action.detach())
                    if isinstance(base_actions, torch.Tensor):
                        base_actions = torch.squeeze(base_actions.detach())
                else:
                    current_action = current_action.detach()
                    if isinstance(base_actions, torch.Tensor):
                        base_actions = base_actions.detach()

                if self.clip_actions:
                    residual_action = rlg_players.rescale_actions(
                        self.actions_low,
                        self.actions_high,
                        torch.clamp(current_action, -1.0, 1.0),
                    )
                else:
                    residual_action = current_action

                if not isinstance(base_actions, torch.Tensor):
                    return residual_action

                base_actions = torch.clamp(base_actions.to(residual_action.device), -1.0, 1.0)
                return torch.cat([base_actions, residual_action], dim=-1)

            rlg_players.PpoPlayerContinuous.get_action = _patched_player_get_action
            rlg_players.PpoPlayerContinuous._maniptrans_residual_concat_patched = True

    runner = Runner()
    runner.load(cfg)
    runner.reset()

    run_args = {
        "train": mode == "train",
        "play": mode == "test",
        "checkpoint": model_file if model_file != "" else None,
        "sigma": None,
    }
    print(f"[rlgames] mode={mode} num_envs={num_envs} device={device}")
    runner.run(run_args)


if __name__ == "__main__":
    main(sys.argv)
