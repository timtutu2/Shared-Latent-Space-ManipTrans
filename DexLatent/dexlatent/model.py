"""VAE model and trainer for cross-embodiment hand latent learning.

Adapted from DexLatent for standalone hands with 6-DOF wrist pass-through.
Wrist DOF (3 pos + 3 axis-angle) replaces the original xarm7 arm and is
passed through the autoencoder unchanged, just like the original.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from dexlatent.kinematics import (
    MultiHandDifferentiableFK,
    solve_inverse_kinematics,
    wrist_dof_to_transform,
)

PINCH_PAIR_DEFAULTS: Tuple[Tuple[int, int], ...] = ((0, 1), (0, 2), (0, 3), (0, 4))


@dataclass
class TrainingConfig:
    device: torch.device = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    )
    wrist_dof: int = 6  # 3 pos + 3 axis-angle, passed through (not encoded)
    latent_dim: int = 32
    hidden_dims: Sequence[int] = (64, 128, 64)
    batch_size: int = 1024
    num_steps: int = 10_000
    learning_rate: float = 2e-3
    rec_weight: float = 1.0
    lambda_dis: float = 2000.0
    lambda_dir: float = 5.0
    lambda_dis_exp: float = 12.0
    lambda_kl: float = 0.001
    pinch_pairs: Sequence[Tuple[int, int]] = field(default_factory=lambda: list(PINCH_PAIR_DEFAULTS))
    pinch_sampling_probability: float = 0.5
    pinch_offset: Tuple[float, float, float] = (0.07, 0.0, -0.08)
    pinch_template_target_noise_std: float = 0.01
    pinch_template_joint_noise_std: float = 0.01
    pinch_template_iterations: int = 100
    pinch_template_learning_rate: float = 0.05
    pinch_template_count: int = 2048
    # Wrist sampling ranges for random training
    wrist_pos_range: float = 0.1   # uniform(-range, +range) meters
    wrist_rot_range: float = 0.5   # uniform(-range, +range) radians
    checkpoint_dir: str = field(
        default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Checkpoints", "dexlatent"))
    )
    checkpoint_interval: int = 500


class HandAutoencoder(nn.Module):
    """Hand VAE with wrist pass-through.

    Input:  [wrist(6), hand(N)]
    Wrist is split off and passed through unchanged.
    Hand is encoded to latent and decoded back.
    Output: [wrist(6), decoded_hand(N)]
    """

    def __init__(self, wrist_dof: int, hand_dof: int, latent_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.wrist_dof = int(wrist_dof)
        self.hand_dof = int(hand_dof)
        self.latent_dim = int(latent_dim)

        # Encoder (hand only)
        self.encoder_backbone = self._make_mlp(self.hand_dof, hidden_dims)
        backbone_out = hidden_dims[-1] if hidden_dims else self.hand_dof
        self.mean_head = nn.Linear(backbone_out, self.latent_dim)
        self.logvar_head = nn.Linear(backbone_out, self.latent_dim)

        # Decoder (hand only)
        self.hand_decoder = self._make_mlp(
            self.latent_dim, hidden_dims, output_dim=self.hand_dof, final_activation=nn.Tanh()
        )

    @staticmethod
    def _make_mlp(
        input_dim: int,
        hidden_dims: Sequence[int],
        *,
        output_dim: Optional[int] = None,
        final_activation: Optional[nn.Module] = None,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        current_dim = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(current_dim, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.ReLU())
            current_dim = width
        if output_dim is not None:
            layers.append(nn.Linear(current_dim, output_dim))
            if final_activation is not None:
                layers.append(final_activation)
        return nn.Sequential(*layers)

    def _split_qpos(self, qpos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split [wrist(6), hand(N)] into (wrist, hand)."""
        return qpos[..., :self.wrist_dof], qpos[..., self.wrist_dof:]

    def encode(self, qpos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (wrist_passthrough, mean, logvar)."""
        wrist, hand = self._split_qpos(qpos)
        features = self.encoder_backbone(hand)
        return wrist, self.mean_head(features), self.logvar_head(features)

    def decode_from_latents(self, wrist: torch.Tensor, latent_hand: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (wrist, decoded_hand)."""
        return wrist, self.hand_decoder(latent_hand)

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = mean + std * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def forward(self, qpos: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Returns (wrist, latent_hand, qpos_wrist, qpos_hand, (mean, logvar))."""
        wrist, mean, logvar = self.encode(qpos)
        if deterministic or not self.training:
            latent_hand = mean
        else:
            latent_hand = self.reparameterize(mean, logvar)
        qpos_wrist, qpos_hand = self.decode_from_latents(wrist, latent_hand)
        return wrist, latent_hand, qpos_wrist, qpos_hand, (mean, logvar)


def compute_pinch_loss(
    source_tips: torch.Tensor,
    target_tips: torch.Tensor,
    pinch_pairs: Sequence[Tuple[int, int]],
    lambda_dis_exp: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(pinch_pairs) == 0:
        empty = torch.zeros(source_tips.shape[0], 0, device=source_tips.device, dtype=source_tips.dtype)
        return empty, empty, empty

    pair_idx = torch.tensor(pinch_pairs, device=source_tips.device, dtype=torch.long)
    i0, i1 = pair_idx[:, 0], pair_idx[:, 1]

    delta_src = source_tips[:, i0, :] - source_tips[:, i1, :]
    delta_tgt = target_tips[:, i0, :] - target_tips[:, i1, :]

    dist_src = torch.linalg.norm(delta_src, dim=-1)
    dist_tgt = torch.linalg.norm(delta_tgt, dim=-1)
    distance_term = torch.square(dist_tgt - dist_src)

    dir_src = F.normalize(delta_src, dim=-1, eps=1e-6)
    dir_tgt = F.normalize(delta_tgt, dim=-1, eps=1e-6)
    direction_term = 1.0 - (dir_src * dir_tgt).sum(dim=-1)

    weight = torch.exp(-lambda_dis_exp * dist_src)
    return distance_term, direction_term, weight


class CrossEmbodimentTrainer:
    """Multi-hand cross-embodiment latent trainer with wrist pass-through."""

    def __init__(self, hand_names: Sequence[str], config: TrainingConfig) -> None:
        self.hand_names: List[str] = list(hand_names)
        self.config = config
        self.wrist_dof = int(config.wrist_dof)

        self.registry = MultiHandDifferentiableFK(hand_names)
        self.hand_models = self.registry.models
        self.hand_dof_per_hand: Dict[str, int] = {n: self.hand_models[n].dof_count() for n in self.hand_names}
        # Full DOF = wrist + hand
        self.dof_per_hand: Dict[str, int] = {n: self.wrist_dof + self.hand_dof_per_hand[n] for n in self.hand_names}
        self.tip_count_per_hand: Dict[str, int] = {n: self.hand_models[n].tip_count() for n in self.hand_names}

        self.autoencoders = nn.ModuleDict({
            name: HandAutoencoder(
                wrist_dof=self.wrist_dof,
                hand_dof=self.hand_dof_per_hand[name],
                latent_dim=self.config.latent_dim,
                hidden_dims=self.config.hidden_dims,
            )
            for name in self.hand_names
        })
        self.autoencoders.to(self.config.device)
        self.optimizer = torch.optim.AdamW(self.autoencoders.parameters(), lr=self.config.learning_rate)

        self.checkpoint_root_dir = os.path.abspath(self.config.checkpoint_dir)
        os.makedirs(self.checkpoint_root_dir, exist_ok=True)

        self._pinch_templates: Dict[str, torch.Tensor] = {}
        self._pinch_points: Dict[str, torch.Tensor] = {}
        self._pinch_pair_cache: Dict[Tuple[str, ...], Tuple[Tuple[int, int], ...]] = {}
        self._neutral_tips: Dict[str, torch.Tensor] = {}

        print(f"[DexLatent] Initialized with hands: {self.hand_names}")
        for n in self.hand_names:
            print(f"  {n}: wrist={self.wrist_dof} + hand={self.hand_dof_per_hand[n]} = {self.dof_per_hand[n]} DOFs, {self.tip_count_per_hand[n]} tips")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_qpos(self, hand_name: str, qpos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split full qpos into (wrist, hand)."""
        return qpos[:, :self.wrist_dof], qpos[:, self.wrist_dof:]

    @staticmethod
    def _merge_qpos(wrist: torch.Tensor, hand: torch.Tensor) -> torch.Tensor:
        return torch.cat([wrist, hand], dim=-1)

    def _wrist_to_transform(self, wrist_dof: torch.Tensor) -> torch.Tensor:
        """Convert wrist DOF to 4x4 transform batch."""
        return wrist_dof_to_transform(wrist_dof)

    def _fk_with_wrist(self, hand_name: str, qpos: torch.Tensor) -> torch.Tensor:
        """Run FK with wrist transform. qpos = [wrist(6), hand(N)]."""
        wrist, hand = self._split_qpos(hand_name, qpos)
        wrist_T = self._wrist_to_transform(wrist)
        return self.hand_models[hand_name].forward(hand, wrist_transform=wrist_T)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu")
        self.autoencoders.load_state_dict(payload["autoencoders"], strict=False)
        for ae in self.autoencoders.values():
            ae.eval()
        print(f"[DexLatent] Loaded checkpoint from {path}")

    def save_checkpoint(self, epoch: int) -> str:
        config_dict = asdict(self.config)
        config_dict["device"] = str(self.config.device)
        config_dict["checkpoint_dir"] = self.checkpoint_dir
        payload = {
            "epoch": epoch,
            "hand_names": self.hand_names,
            "config": config_dict,
            "autoencoders": self.autoencoders.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pt")
        torch.save(payload, path)
        return path

    # ------------------------------------------------------------------
    # Pinch pairs
    # ------------------------------------------------------------------

    def pinch_pairs_for_hand(self, hand_name: str) -> Tuple[Tuple[int, int], ...]:
        key = (hand_name,)
        if key in self._pinch_pair_cache:
            return self._pinch_pair_cache[key]
        limit = self.tip_count_per_hand[hand_name]
        filtered = tuple(p for p in self.config.pinch_pairs if p[0] < limit and p[1] < limit)
        self._pinch_pair_cache[key] = filtered
        return filtered

    def shared_pinch_pairs(self, hand_a: str, hand_b: str) -> Tuple[Tuple[int, int], ...]:
        key = tuple(sorted((hand_a, hand_b)))
        if key in self._pinch_pair_cache:
            return self._pinch_pair_cache[key]
        limit = min(self.tip_count_per_hand[hand_a], self.tip_count_per_hand[hand_b])
        filtered = tuple(p for p in self.config.pinch_pairs if p[0] < limit and p[1] < limit)
        self._pinch_pair_cache[key] = filtered
        return filtered

    # ------------------------------------------------------------------
    # Pinch template generation (hand-only, no wrist)
    # ------------------------------------------------------------------

    def _cache_pinch_templates(self) -> None:
        template_count = int(self.config.pinch_template_count)
        noise_std = self.config.pinch_template_target_noise_std
        base_offset = torch.tensor(self.config.pinch_offset, dtype=torch.float32)

        for hand_name in self.hand_names:
            model = self.hand_models[hand_name]
            dof = model.dof_count()
            neutral = model.angles_to_normalized(torch.zeros(dof, dtype=torch.float32))
            tips = model.forward(neutral)  # aligned frame
            neutral_tips = tips.detach()
            self._neutral_tips[hand_name] = neutral_tips

            # Pinch point = offset from origin in aligned frame
            pinch_base = base_offset.clone()

            target_sets: List[torch.Tensor] = []
            pinch_points: List[torch.Tensor] = []
            tip_count = self.tip_count_per_hand[hand_name]

            for finger_idx in range(1, tip_count):
                base_target = neutral_tips.clone()
                base_target[0] = pinch_base
                base_target[finger_idx] = pinch_base
                target_sets.append(base_target)
                pinch_points.append(pinch_base.clone())

            if not target_sets:
                self._pinch_templates[hand_name] = torch.zeros(0, dof, dtype=torch.float32)
                self._pinch_points[hand_name] = torch.zeros(0, 3, dtype=torch.float32)
                continue

            base_targets = torch.stack(target_sets, dim=0)
            pp_tensor = torch.stack(pinch_points, dim=0)
            repeats = math.ceil(template_count / base_targets.shape[0])
            tiled_targets = base_targets.repeat_interleave(repeats, dim=0)[:template_count]
            tiled_points = pp_tensor.repeat_interleave(repeats, dim=0)[:template_count]

            noise = torch.randn_like(tiled_targets) * noise_std
            noise[:, 0, :] = 0.0
            perturbed = tiled_targets + noise
            perturbed[:, 0, :] = tiled_points

            history = solve_inverse_kinematics(
                model=model, target_positions=perturbed,
                iterations=self.config.pinch_template_iterations,
                learning_rate=self.config.pinch_template_learning_rate,
            )
            self._pinch_templates[hand_name] = history[-1].detach().cpu()
            self._pinch_points[hand_name] = tiled_points.detach().cpu()

        print("[DexLatent] Pinch templates cached.")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_wrist(self, batch_size: int) -> torch.Tensor:
        """Sample random wrist DOF: [pos(3), axis-angle(3)]."""
        device = self.config.device
        pos = torch.empty(batch_size, 3, device=device).uniform_(
            -self.config.wrist_pos_range, self.config.wrist_pos_range
        )
        rot = torch.empty(batch_size, 3, device=device).uniform_(
            -self.config.wrist_rot_range, self.config.wrist_rot_range
        )
        return torch.cat([pos, rot], dim=-1)  # (B, 6)

    def _sample_pinch_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        device = self.config.device
        wrist = self._sample_wrist(batch_size)
        result: Dict[str, torch.Tensor] = {}
        for name in self.hand_names:
            templates = self._pinch_templates[name]
            if templates.shape[0] == 0:
                result[name] = torch.empty(0, self.dof_per_hand[name], device=device)
                continue
            idx = torch.randint(0, templates.shape[0], (batch_size,))
            hand = templates[idx].to(device=device)
            noise = torch.randn_like(hand) * self.config.pinch_template_joint_noise_std
            hand = torch.clamp(hand + noise, -1.0, 1.0)
            result[name] = self._merge_qpos(wrist, hand)
        return result

    def _sample_uniform(self) -> Dict[str, torch.Tensor]:
        device = self.config.device
        bs = self.config.batch_size
        wrist = self._sample_wrist(bs)
        return {
            name: self._merge_qpos(
                wrist,
                torch.empty(bs, self.hand_dof_per_hand[name], device=device).uniform_(-1.0, 1.0),
            )
            for name in self.hand_names
        }

    def _sample_training_batch(self) -> Dict[str, torch.Tensor]:
        bs = self.config.batch_size
        pinch_count = int(round(bs * self.config.pinch_sampling_probability))
        uniform_count = bs - pinch_count
        uniform = self._sample_uniform()
        if pinch_count <= 0:
            return uniform
        pinch = self._sample_pinch_batch(pinch_count)
        mixed: Dict[str, torch.Tensor] = {}
        for name in self.hand_names:
            if uniform_count <= 0:
                batch = pinch[name]
            else:
                batch = torch.cat([uniform[name], pinch[name]], dim=0)
            mixed[name] = batch[torch.randperm(batch.shape[0], device=self.config.device)]
        return mixed

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, float]:
        self.optimizer.zero_grad()
        batch = self._sample_training_batch()

        latents: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}  # (wrist, hand_latent)
        latent_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        source_tips: Dict[str, torch.Tensor] = {}
        rec_loss = torch.tensor(0.0, device=self.config.device)

        for name in self.hand_names:
            qpos = batch[name]
            _, hand_gt = self._split_qpos(name, qpos)
            wrist, latent_hand, _, hand_pred, stats = self.autoencoders[name](qpos)
            latents[name] = (wrist, latent_hand)
            latent_stats[name] = stats
            rec_loss = rec_loss + F.mse_loss(hand_pred, hand_gt)
            source_tips[name] = self._fk_with_wrist(name, qpos)

        rec_loss = rec_loss / len(self.hand_names) * self.config.rec_weight

        # Cross-embodiment pinch loss
        pinch_dis_total = torch.tensor(0.0, device=self.config.device)
        pinch_dir_total = torch.tensor(0.0, device=self.config.device)
        exp_w_total = torch.tensor(0.0, device=self.config.device)
        pair_count = 0

        for src_name in self.hand_names:
            src_fk = source_tips[src_name]
            src_wrist, src_latent = latents[src_name]
            for tgt_name in self.hand_names:
                if tgt_name == src_name:
                    continue
                pp = self.shared_pinch_pairs(src_name, tgt_name)
                if not pp:
                    continue
                # Decode target hand from source latent, keep source wrist
                tgt_wrist, tgt_hand = self.autoencoders[tgt_name].decode_from_latents(src_wrist, src_latent)
                tgt_qpos = self._merge_qpos(tgt_wrist.detach(), tgt_hand)
                tgt_fk = self._fk_with_wrist(tgt_name, tgt_qpos)
                dist_term, dir_term, exp_w = compute_pinch_loss(src_fk, tgt_fk, pp, self.config.lambda_dis_exp)
                pair_count += 1
                pinch_dis_total = pinch_dis_total + (dist_term * exp_w).mean()
                pinch_dir_total = pinch_dir_total + (dir_term * exp_w).mean()
                exp_w_total = exp_w_total + exp_w.mean()

        if pair_count > 0:
            pinch_dis = pinch_dis_total / pair_count
            pinch_dir = pinch_dir_total / pair_count
            exp_w_mean = exp_w_total / pair_count
        else:
            pinch_dis = pinch_dir = exp_w_mean = torch.tensor(0.0, device=self.config.device)

        # KL divergence
        kl_total = torch.tensor(0.0, device=self.config.device)
        for mean, logvar in latent_stats.values():
            kl_total = kl_total + (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)).mean()
        kl_loss = kl_total / len(self.hand_names)

        total = rec_loss + self.config.lambda_dis * pinch_dis + self.config.lambda_dir * pinch_dir + self.config.lambda_kl * kl_loss
        total.backward()
        self.optimizer.step()

        return {
            "loss_total": float(total.detach()),
            "loss_rec": float(rec_loss.detach()),
            "loss_pinch_dis": float(pinch_dis.detach()) * self.config.lambda_dis,
            "loss_pinch_dir": float(pinch_dir.detach()) * self.config.lambda_dir,
            "loss_kl": float(kl_loss.detach()) * self.config.lambda_kl,
            "exp_dis": float(exp_w_mean.detach()),
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def _init_checkpoint_session_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.checkpoint_root_dir, timestamp)
        suffix = 1
        while os.path.exists(run_dir):
            run_dir = os.path.join(self.checkpoint_root_dir, f"{timestamp}_{suffix:02d}")
            suffix += 1
        os.makedirs(run_dir, exist_ok=False)
        return run_dir

    def train(self) -> List[Dict[str, float]]:
        self.checkpoint_dir = self._init_checkpoint_session_dir()
        print(f"[DexLatent] Training {self.config.num_steps} steps, checkpoints -> {self.checkpoint_dir}")
        self._cache_pinch_templates()

        for ae in self.autoencoders.values():
            ae.train()

        history: List[Dict[str, float]] = []
        for step_idx in range(self.config.num_steps):
            metrics = self.step()
            history.append(metrics)
            epoch = step_idx + 1
            print(
                f"Step {epoch:04d} | "
                f"total={metrics['loss_total']:.4f} "
                f"rec={metrics['loss_rec']:.4f} "
                f"pinch_dis={metrics['loss_pinch_dis']:.4f} "
                f"pinch_dir={metrics['loss_pinch_dir']:.4f} "
                f"exp={metrics['exp_dis']:.4f} "
                f"kl={metrics['loss_kl']:.4f}"
            )
            if self.config.checkpoint_interval > 0 and epoch % self.config.checkpoint_interval == 0:
                path = self.save_checkpoint(epoch)
                print(f"  -> Saved {path}")

        return history
