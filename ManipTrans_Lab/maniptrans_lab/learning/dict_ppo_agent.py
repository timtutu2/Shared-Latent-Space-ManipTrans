"""
Dict-obs PPOAgent for ManipTrans_Lab.

Extends MimicKit's PPOAgent with minimal changes to support a dict observation
space (keys: proprioception, privileged, target). Everything else (PPO updates,
return computation, optimizer setup) is inherited unchanged.

Dict obs are stored in the experience buffer as flat per-key streams
(keyed as "obs__proprioception", "next_obs__proprioception", …). The
DictPPOModel consumes them as a dict again at loss time.
"""
import gymnasium.spaces as spaces
import numpy as np
import torch

import envs.base_env as base_env
import learning.base_agent as base_agent
import learning.dict_normalizer as dict_normalizer
import learning.dict_ppo_model as dict_ppo_model
import learning.ppo_agent as ppo_agent
import learning.rl_util as rl_util
import util.mp_util as mp_util
import util.torch_util as torch_util


OBS_PREFIX = "obs__"
NEXT_OBS_PREFIX = "next_obs__"


class DictPPOAgent(ppo_agent.PPOAgent):
    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        return

    # -- build dict model / dict normalizer ------------------------------- #
    def _build_model(self, config):
        model_config = config["model"]
        self._model = dict_ppo_model.DictPPOModel(model_config, self._env)
        return

    def _build_normalizers(self):
        obs_space = self._env.get_obs_space()
        if isinstance(obs_space, spaces.Dict):
            self._obs_norm = dict_normalizer.DictNormalizer(
                obs_space, device=self._device, clip=10.0)
        else:
            # fall back to flat Normalizer (MimicKit default)
            super()._build_normalizers()
            return
        self._a_norm = self._build_action_normalizer()
        return

    # -- dict-aware recording of experience ------------------------------- #
    def _record_data_pre_step(self, obs, info, action, action_info):
        if isinstance(obs, dict):
            for k, v in obs.items():
                self._exp_buffer.record(OBS_PREFIX + k, v)
            self._exp_buffer.record("action", action)
            if self._need_normalizer_update():
                self._obs_norm.record(obs)
        else:
            super()._record_data_pre_step(obs, info, action, action_info)

        self._exp_buffer.record("a_logp", action_info["a_logp"])
        self._exp_buffer.record("rand_action_mask", action_info["rand_action_mask"])
        return

    def _record_data_post_step(self, next_obs, r, done, next_info):
        if isinstance(next_obs, dict):
            for k, v in next_obs.items():
                self._exp_buffer.record(NEXT_OBS_PREFIX + k, v)
            self._exp_buffer.record("reward", r)
            self._exp_buffer.record("done", done)
        else:
            super()._record_data_post_step(next_obs, r, done, next_info)
        return

    # -- PPO updates: re-assemble dict from per-key buffers --------------- #
    def _obs_dict_from_buffer(self, prefix):
        out = {}
        for name, buf in self._exp_buffer._flat_buffers.items():
            if name.startswith(prefix):
                k = name[len(prefix):]
                out[k] = self._exp_buffer.get_data(name)
        return out

    def _decide_action(self, obs, info):
        if isinstance(obs, dict):
            norm_obs = self._obs_norm.normalize(obs)
        else:
            norm_obs = self._obs_norm.normalize(obs)
        dist = self._model.eval_actor(norm_obs)

        if self._mode == base_agent.AgentMode.TRAIN:
            norm_a_rand = dist.sample()
            norm_a_mode = dist.mode
            exp_prob = self._get_exp_prob()
            exp_prob = torch.full([norm_a_rand.shape[0], 1], exp_prob,
                                  device=self._device, dtype=torch.float)
            rand_action_mask = torch.bernoulli(exp_prob)
            norm_a = torch.where(rand_action_mask == 1.0, norm_a_rand, norm_a_mode)
            rand_action_mask = rand_action_mask.squeeze(-1)
        else:
            norm_a = dist.mode
            rand_action_mask = torch.zeros_like(norm_a[..., 0])

        norm_a_logp = dist.log_prob(norm_a).detach()
        norm_a = norm_a.detach()
        a = self._a_norm.unnormalize(norm_a)
        return a, {"a_logp": norm_a_logp, "rand_action_mask": rand_action_mask}

    # -- value & advantage: flatten dict obs to keep PPOAgent._build_train_data clean --
    def _build_train_data(self):
        self.eval()

        # re-assemble dict obs
        obs = self._obs_dict_from_buffer(OBS_PREFIX)
        next_obs = self._obs_dict_from_buffer(NEXT_OBS_PREFIX)

        r = self._exp_buffer.get_data("reward")
        done = self._exp_buffer.get_data("done")
        rand_action_mask = self._exp_buffer.get_data("rand_action_mask")

        norm_next_obs = self._obs_norm.normalize(next_obs)
        next_vals = self._eval_critic_batched(
            norm_next_obs, self._critic_eval_batch_size
        ).squeeze(-1).detach()

        succ_val = self._compute_succ_val()
        fail_val = self._compute_fail_val()
        next_vals[done == base_env.DoneFlags.SUCC.value] = succ_val
        next_vals[done == base_env.DoneFlags.FAIL.value] = fail_val

        new_vals = rl_util.compute_td_lambda_return(
            r, next_vals, done, self._discount, self._td_lambda)

        norm_obs = self._obs_norm.normalize(obs)
        vals = self._eval_critic_batched(
            norm_obs, self._critic_eval_batch_size
        ).squeeze(-1).detach()
        adv = new_vals - vals

        rand_action_mask_flat = (rand_action_mask == 1.0).flatten()
        rand_action_adv = adv.flatten()[rand_action_mask_flat]
        adv_mean, adv_std = mp_util.calc_mean_std(rand_action_adv)
        norm_adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-5)
        norm_adv = torch.clamp(norm_adv, -self._norm_adv_clip, self._norm_adv_clip)

        self._exp_buffer.set_data("tar_val", new_vals)
        self._exp_buffer.set_data("adv", norm_adv)

        return {"adv_mean": adv_mean, "adv_std": adv_std}

    def _eval_critic_batched(self, obs, batch_size):
        """
        Dict-aware minibatch value evaluation.

        The MimicKit util.torch_util.eval_minibatch(fn, inputs, bs) does
            fn(**inputs)
        which kwargs-unpacks its input dict — that clashes with our
        eval_critic(obs_dict) signature. Here we:
          * flatten any leading batch dims of each dict value into dim 0,
          * iterate in chunks of batch_size along that flat dim,
          * gather per-batch outputs and reshape back to the original
            leading dims of obs.
        """
        if not isinstance(obs, dict):
            return torch_util.eval_minibatch(
                self._model.eval_critic, {"obs": obs}, batch_size)

        ref = next(iter(obs.values()))
        lead_shape = ref.shape[:-1] if ref.dim() >= 2 else ref.shape
        # flatten all leading dims so we can batch along a single axis
        flat_obs = {k: v.reshape(-1, *v.shape[len(lead_shape):]) for k, v in obs.items()}
        n = next(iter(flat_obs.values())).shape[0]

        if batch_size is None or batch_size <= 0:
            out = self._model.eval_critic(flat_obs)
        else:
            outs = []
            for i in range(0, n, batch_size):
                chunk = {k: v[i:i + batch_size] for k, v in flat_obs.items()}
                outs.append(self._model.eval_critic(chunk))
            out = torch.cat(outs, dim=0)

        # out is [n, out_dim]; reshape back to (*lead_shape, out_dim)
        return out.reshape(*lead_shape, *out.shape[1:])

    def _compute_critic_loss(self, batch):
        # reassemble dict from per-key batch entries
        obs = {k[len(OBS_PREFIX):]: v for k, v in batch.items() if k.startswith(OBS_PREFIX)}
        norm_obs = self._obs_norm.normalize(obs) if obs else self._obs_norm.normalize(batch["obs"])
        tar_val = batch["tar_val"]
        pred = self._model.eval_critic(norm_obs).squeeze(-1)
        loss = torch.mean(torch.square(tar_val - pred))
        return {"critic_loss": loss}

    def _compute_actor_loss(self, batch):
        obs = {k[len(OBS_PREFIX):]: v for k, v in batch.items() if k.startswith(OBS_PREFIX)}
        if obs:
            norm_obs = self._obs_norm.normalize(obs)
            rand_action_mask = (batch["rand_action_mask"] == 1.0)
            norm_obs = {k: v[rand_action_mask] for k, v in norm_obs.items()}
        else:
            norm_obs = self._obs_norm.normalize(batch["obs"])
            rand_action_mask = (batch["rand_action_mask"] == 1.0)
            norm_obs = norm_obs[rand_action_mask]

        norm_a = self._a_norm.normalize(batch["action"])[rand_action_mask]
        old_a_logp = batch["a_logp"][rand_action_mask]
        adv = batch["adv"][rand_action_mask]

        dist = self._model.eval_actor(norm_obs)
        a_logp = dist.log_prob(norm_a)

        a_ratio = torch.exp(a_logp - old_a_logp)
        loss0 = adv * a_ratio
        loss1 = adv * torch.clamp(a_ratio, 1.0 - self._ppo_clip_ratio, 1.0 + self._ppo_clip_ratio)
        actor_loss = -torch.mean(torch.minimum(loss0, loss1))

        clip_frac = (torch.abs(a_ratio - 1.0) > self._ppo_clip_ratio).float().mean()
        imp_ratio = a_ratio.mean()

        info = {"actor_loss": actor_loss, "clip_frac": clip_frac.detach(),
                "imp_ratio": imp_ratio.detach()}

        if self._action_bound_weight != 0:
            bound = self._compute_action_bound_loss(dist)
            if bound is not None:
                bound = bound.mean()
                actor_loss = actor_loss + self._action_bound_weight * bound
                info["action_bound_loss"] = bound.detach()

        if self._action_entropy_weight != 0:
            ent = dist.entropy().mean()
            actor_loss = actor_loss - self._action_entropy_weight * ent
            info["action_entropy"] = ent.detach()

        if self._action_reg_weight != 0:
            reg = dist.param_reg().mean()
            actor_loss = actor_loss + self._action_reg_weight * reg
            info["action_reg_loss"] = reg.detach()

        info["actor_loss"] = actor_loss
        return info
