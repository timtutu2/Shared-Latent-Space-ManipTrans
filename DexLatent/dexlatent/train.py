"""Training entry point for cross-embodiment hand latent learning.

Usage:
    cd ManipTransDexLatent
    python -m dexlatent.train
    python -m dexlatent.train --hands shadow_right inspire_right mano_right --num_steps 5000
"""

from __future__ import annotations

import argparse
import sys

import torch

from dexlatent.model import CrossEmbodimentTrainer, TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cross-embodiment hand latent model.")
    parser.add_argument(
        "--hands",
        nargs="+",
        default=["mano_right", "shadow_right", "inspire_right"],
        help="Hand names to train (default: mano_right shadow_right inspire_right)",
    )
    parser.add_argument("--num_steps", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--pinch_template_count", type=int, default=2048)
    parser.add_argument("--pinch_template_iterations", type=int, default=100)
    parser.add_argument("--lambda_kl", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    config = TrainingConfig(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        checkpoint_interval=args.checkpoint_interval,
        pinch_template_count=args.pinch_template_count,
        pinch_template_iterations=args.pinch_template_iterations,
        lambda_kl=args.lambda_kl,
    )

    trainer = CrossEmbodimentTrainer(args.hands, config)
    trainer.train()


if __name__ == "__main__":
    main()
