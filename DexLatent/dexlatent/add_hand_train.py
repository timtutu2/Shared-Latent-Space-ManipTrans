"""Train a new hand autoencoder while keeping existing hands frozen.

Based on DexLatent's shadow_train.py. Loads a pretrained checkpoint,
freezes existing hand autoencoders, and trains only the new hand.

Usage:
    cd ManipTransDexLatent

    python -m dexlatent.add_hand_train \
        --ckpt Checkpoints/dexlatent/dexlatent_arti_sha_ins/checkpoint_epoch_5000.pt \
        --new_hand allegro_right \
        --num_steps 5000
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import torch

from dexlatent.model import CrossEmbodimentTrainer, TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train new hand encoder/decoder with frozen existing hands."
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Existing checkpoint with trained hands.")
    parser.add_argument("--new_hand", type=str, required=True,
                        help="New hand name to train (e.g. allegro_right)")
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--pinch_template_count", type=int, default=2048)
    parser.add_argument("--pinch_template_iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load existing checkpoint to get hand list
    ckpt_path = os.path.abspath(args.ckpt)
    print(f"[DexLatent] Loading base checkpoint: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu")
    existing_hands = payload["hand_names"]
    print(f"  Existing hands: {existing_hands}")

    if args.new_hand in existing_hands:
        print(f"  WARNING: {args.new_hand} already in checkpoint, will retrain it")

    # All hands = existing + new
    all_hands = list(existing_hands)
    if args.new_hand not in all_hands:
        all_hands.append(args.new_hand)

    config = TrainingConfig(
        num_steps=args.num_steps,
        checkpoint_interval=args.checkpoint_interval,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        pinch_template_count=args.pinch_template_count,
        pinch_template_iterations=args.pinch_template_iterations,
    )

    # Build trainer with all hands
    trainer = CrossEmbodimentTrainer(all_hands, config)

    # Load existing weights (new hand will be skipped via strict=False)
    trainer.load_checkpoint(ckpt_path)

    # Freeze existing hands
    for name in existing_hands:
        if name == args.new_hand:
            continue
        for param in trainer.autoencoders[name].parameters():
            param.requires_grad = False
        trainer.autoencoders[name].eval()
    print(f"  Frozen: {[n for n in existing_hands if n != args.new_hand]}")

    # Set new hand to train mode
    trainer.autoencoders[args.new_hand].train()
    new_params = list(trainer.autoencoders[args.new_hand].parameters())
    trainable = sum(p.numel() for p in new_params)
    total = sum(p.numel() for p in trainer.autoencoders.parameters())
    print(f"  Training: {args.new_hand} ({trainable:,} params / {total:,} total)")
    print(f"  DOFs: wrist={trainer.config.wrist_dof} + hand={trainer.hand_dof_per_hand[args.new_hand]} = {trainer.dof_per_hand[args.new_hand]}")
    print(f"  Tips: {trainer.tip_count_per_hand[args.new_hand]}")

    # Rebuild optimizer with only new hand params
    trainer.optimizer = torch.optim.AdamW(new_params, lr=config.learning_rate)

    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(trainer.checkpoint_root_dir, f"{args.new_hand}_{timestamp}")
    os.makedirs(ckpt_dir, exist_ok=True)
    trainer.checkpoint_dir = ckpt_dir

    # Cache pinch templates
    print("[DexLatent] Generating pinch templates...")
    trainer._cache_pinch_templates()

    # Training loop
    print(f"\n[DexLatent] Training {args.new_hand} for {config.num_steps} steps...")
    print(f"  Checkpoints -> {ckpt_dir}\n")

    for step_idx in range(config.num_steps):
        metrics = trainer.step()
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
        if config.checkpoint_interval > 0 and epoch % config.checkpoint_interval == 0:
            path = trainer.save_checkpoint(epoch)
            print(f"  -> Saved {path}")

    print(f"\nDone. Checkpoints in: {ckpt_dir}")


if __name__ == "__main__":
    main()
