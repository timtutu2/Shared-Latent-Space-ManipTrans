"""Visualize cross-embodiment hand latent decoding in Isaac Lab.

Uses ManipTransDexLatent's engine abstraction (build_engine) for rendering.
Shows source pkl original + decoded hands side by side.

Usage:
    cd ManipTransDexLatent

    # From GRAB demo pkl
    python -m dexlatent.visualize_isaaclab \
        --ckpt Checkpoints/dexlatent/<run>/checkpoint_epoch_XXXX.pt \
        --source mano_right \
        --motion_pkl data/motion_data/grab_demo/retargeting/mano2artimano_rh/102_sv_dict.pkl

    # Random mode (no pkl)
    python -m dexlatent.visualize_isaaclab \
        --ckpt <path> --source mano_right --num_frames 200
"""

from __future__ import annotations

import argparse
import os
import sys
import pickle
import time
from typing import Dict, List

import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRotation

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "mainptrans"))

from dexlatent.kinematics import HAND_CONFIGS, ALIGNMENT_ROTATIONS
from dexlatent.model import CrossEmbodimentTrainer, TrainingConfig


def denormalize_qpos(model, normalized: torch.Tensor) -> torch.Tensor:
    """Convert normalized [-1,1] qpos to real joint angles (radians)."""
    clipped = torch.clamp(normalized, -1.0, 1.0)
    lower = model._lower.to(device=clipped.device, dtype=clipped.dtype)
    upper = model._upper.to(device=clipped.device, dtype=clipped.dtype)
    return (clipped + 1.0) * 0.5 * (upper - lower) + lower


def build_articulation_cfg(hand_name: str) -> dict:
    """Build articulation_cfg matching ManipTransDexLatent env configs."""
    if hand_name.startswith("shadow"):
        return {
            "joint_names_expr": [
                "r_(FF|MF|RF|LF|TH)J(3|2|1|0)",
                "r_(LF|TH)J4",
            ] if "right" in hand_name else [
                "l_(FF|MF|RF|LF|TH)J(3|2|1|0)",
                "l_(LF|TH)J4",
            ],
            "effort_limit": None, "velocity_limit": None,
            "effort_limit_sim": {".*": 2.0}, "velocity_limit_sim": None,
            "stiffness": {".*": 500.0}, "damping": {".*": 30.0},
            "armature": None, "friction": None, "dynamic_friction": None, "viscous_friction": None,
        }
    elif hand_name.startswith("inspire"):
        return {
            "joint_names_expr": [
                "R_(index|middle|pinky|ring)_(proximal|intermediate)_joint",
                "R_thumb_proximal_(yaw|pitch)_joint",
                "R_thumb_intermediate_joint",
                "R_thumb_distal_joint",
            ],
            "effort_limit": None, "velocity_limit": None,
            "effort_limit_sim": {".*": 2.0}, "velocity_limit_sim": None,
            "stiffness": {".*": 500.0}, "damping": {".*": 30.0},
            "armature": None, "friction": None, "dynamic_friction": None, "viscous_friction": None,
        }
    elif hand_name.startswith("mano"):
        prefix = "r_" if "right" in hand_name else "l_"
        return {
            "joint_names_expr": [f"{prefix}.*"],
            "effort_limit": None, "velocity_limit": None,
            "effort_limit_sim": {".*": 2.0}, "velocity_limit_sim": None,
            "stiffness": {".*": 500.0}, "damping": {".*": 30.0},
            "armature": None, "friction": None, "dynamic_friction": None, "viscous_friction": None,
        }
    elif hand_name.startswith("allegro"):
        return {
            "joint_names_expr": ["r_joint_.*"],
            "effort_limit": None, "velocity_limit": None,
            "effort_limit_sim": {".*": 10.0}, "velocity_limit_sim": None,
            "stiffness": {".*": 500.0}, "damping": {".*": 30.0},
            "armature": None, "friction": None, "dynamic_friction": None, "viscous_friction": None,
        }
    else:
        raise ValueError(f"Unknown hand: {hand_name}")


def build_all_joints_to_engine_reorder(eng, obj_id: int, all_joint_names: List[str]) -> torch.Tensor:
    """Build gather index: engine_cmd = fk_angles[:, reorder].

    reorder[engine_idx] = fk_idx, so for each engine position we pick
    the correct FK joint value.
    """
    obj = eng._objs[obj_id]
    dof_order_sim2common = eng._dof_order_sim2common[obj_id].tolist()

    engine_joint_names = list(obj.joint_names)
    common_joint_names = [engine_joint_names[dof_order_sim2common[i]]
                          for i in range(len(dof_order_sim2common))]

    reorder = []
    for eng_name in common_joint_names:
        found = False
        for j, fk_name in enumerate(all_joint_names):
            if fk_name == eng_name:
                reorder.append(j)
                found = True
                break
        if not found:
            print(f"  WARNING: Engine joint '{eng_name}' not found in FK!")
            reorder.append(0)

    return torch.tensor(reorder, dtype=torch.long)


# ------------------------------------------------------------------
# Motion loading & trajectory generation
# ------------------------------------------------------------------

def load_motion_pkl(pkl_path: str, trainer: CrossEmbodimentTrainer, source_hand: str) -> torch.Tensor:
    """Load GRAB demo pkl -> full qpos [wrist(6), hand(N)].

    If pkl has more DOFs than the model (e.g. pkl has 12 but model has 6
    due to mimic exclusion), extract only the independent DOF columns.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    dof_pos = torch.tensor(data["opt_dof_pos"], dtype=torch.float32)
    wrist_pos = torch.tensor(data["opt_wrist_pos"], dtype=torch.float32)
    wrist_rot = torch.tensor(data["opt_wrist_rot"], dtype=torch.float32)
    fk_model = trainer.hand_models[source_hand]

    T = dof_pos.shape[0]
    pkl_dof = dof_pos.shape[1]
    model_dof = fk_model.dof_count()

    if pkl_dof == model_dof:
        hand_angles = dof_pos
    elif pkl_dof > model_dof:
        from dexlatent.kinematics import load_urdf_silent, HAND_CONFIGS
        urdf = load_urdf_silent(str(HAND_CONFIGS[source_hand]["urdf_path"]))
        all_revolute = []
        independent_indices = []
        for joint in urdf.joints:
            if joint.type == "revolute":
                is_mimic = joint.mimic is not None
                all_revolute.append((joint.name, is_mimic))
                if not is_mimic:
                    independent_indices.append(len(all_revolute) - 1)
        hand_angles = dof_pos[:, independent_indices]
    else:
        raise ValueError(f"DOF mismatch: pkl has {pkl_dof} but model has {model_dof}")

    hand_normalized = torch.clamp(fk_model.angles_to_normalized(hand_angles), -1.0, 1.0)
    wrist_dof = torch.cat([wrist_pos, wrist_rot], dim=-1)
    full_qpos = torch.cat([wrist_dof, hand_normalized], dim=-1)
    return full_qpos


def generate_smooth_trajectory(
    trainer: CrossEmbodimentTrainer, source_hand: str,
    num_frames: int = 200, num_keyframes: int = 10,
) -> Dict[str, torch.Tensor]:
    device = trainer.config.device
    dof = trainer.dof_per_hand[source_hand]

    keyframes = torch.empty(num_keyframes, dof, device=device).uniform_(-0.8, 0.8)
    keyframes = torch.cat([keyframes, keyframes[:1]], dim=0)

    frames_per_seg = num_frames // num_keyframes
    all_frames = []
    for i in range(num_keyframes):
        for t in range(frames_per_seg):
            alpha = t / frames_per_seg
            all_frames.append((1 - alpha) * keyframes[i] + alpha * keyframes[i + 1])
    return _encode_decode_all(trainer, source_hand, torch.stack(all_frames, dim=0))


def generate_pkl_trajectory(
    trainer: CrossEmbodimentTrainer, source_hand: str, pkl_path: str,
) -> Dict[str, torch.Tensor]:
    source_qpos = load_motion_pkl(pkl_path, trainer, source_hand)
    return _encode_decode_all(trainer, source_hand, source_qpos)


def _encode_decode_all(
    trainer: CrossEmbodimentTrainer, source_hand: str, source_qpos: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Encode source [wrist(6), hand(N)] -> decode to all hands."""
    wrist_dof = trainer.config.wrist_dof

    with torch.no_grad():
        wrist, mean, _ = trainer.autoencoders[source_hand].encode(source_qpos)

    decoded: Dict[str, torch.Tensor] = {}
    for hand_name in trainer.hand_names:
        with torch.no_grad():
            _, hand_decoded = trainer.autoencoders[hand_name].decode_from_latents(wrist, mean)
        decoded[hand_name] = torch.cat([wrist, hand_decoded], dim=-1).cpu()

    decoded[f"{source_hand}_source"] = source_qpos.cpu()

    # --- Latent comparison: re-encode each decoded hand ---
    print("\n" + "=" * 70)
    print("  LATENT COMPARISON (32D)")
    print("=" * 70)

    source_latent = mean.cpu()  # (T, 32)
    print("\n  Source latent ({})  mean={:.4f}  std={:.4f}  range=[{:.4f}, {:.4f}]".format(
        source_hand, source_latent.mean().item(), source_latent.std().item(),
        source_latent.min().item(), source_latent.max().item()))

    for hand_name in trainer.hand_names:
        # Re-encode: decoded hand joints → that hand's encoder → latent
        dec_qpos = decoded[hand_name]
        dec_hand = dec_qpos[:, wrist_dof:]
        dummy_wrist = torch.zeros(dec_hand.shape[0], wrist_dof)
        re_input = torch.cat([dummy_wrist, dec_hand], dim=-1)
        with torch.no_grad():
            _, re_mean, _ = trainer.autoencoders[hand_name].encode(re_input)
        re_latent = re_mean.cpu()

        # Compare
        l2_per_frame = torch.norm(source_latent - re_latent, dim=-1)  # (T,)
        cos_sim = torch.nn.functional.cosine_similarity(source_latent, re_latent, dim=-1)

        print("\n  {} encoder(decoded) vs source latent:".format(hand_name))
        print("    re-encoded  mean={:.4f}  std={:.4f}  range=[{:.4f}, {:.4f}]".format(
            re_latent.mean().item(), re_latent.std().item(),
            re_latent.min().item(), re_latent.max().item()))
        print("    L2 distance:  mean={:.4f}  max={:.4f}  min={:.4f}".format(
            l2_per_frame.mean().item(), l2_per_frame.max().item(), l2_per_frame.min().item()))
        print("    Cosine sim:   mean={:.4f}  max={:.4f}  min={:.4f}".format(
            cos_sim.mean().item(), cos_sim.max().item(), cos_sim.min().item()))

    # --- Double decode & cross-decode ---
    print("\n  " + "-" * 50)
    print("  DOUBLE DECODE & CROSS-DECODE ANALYSIS")
    print("  " + "-" * 50)

    # Source ground truth tips
    src_hand_part = source_qpos[:, wrist_dof:]
    with torch.no_grad():
        source_fk_model = trainer.hand_models[source_hand]
        source_gt_tips = source_fk_model.forward(src_hand_part)

    for hand_name in trainer.hand_names:
        dec_qpos = decoded[hand_name]
        dec_hand = dec_qpos[:, wrist_dof:]
        fk_model = trainer.hand_models[hand_name]
        dummy_wrist = torch.zeros(dec_hand.shape[0], wrist_dof)

        # Path A: z → hand decoder → tips
        with torch.no_grad():
            path_a_tips = fk_model.forward(dec_hand)

        # Re-encode: hand decoder output → hand encoder → z'
        re_input = torch.cat([dummy_wrist, dec_hand], dim=-1)
        with torch.no_grad():
            _, z_prime, _ = trainer.autoencoders[hand_name].encode(re_input)

        # Path B: z' → same hand decoder → tips
        with torch.no_grad():
            _, re_decoded_hand = trainer.autoencoders[hand_name].decode_from_latents(dummy_wrist, z_prime)
            re_decoded_hand = torch.clamp(re_decoded_hand, -1.0, 1.0)
            path_b_tips = fk_model.forward(re_decoded_hand)

        n_tips = min(path_a_tips.shape[1], path_b_tips.shape[1])
        ab_diff = torch.norm(path_a_tips[:, :n_tips] - path_b_tips[:, :n_tips], dim=-1) * 1000

        # Path C: z' → SOURCE decoder → source tips (cross-decode with re-encoded latent)
        with torch.no_grad():
            _, cross_decoded_hand = trainer.autoencoders[source_hand].decode_from_latents(dummy_wrist, z_prime)
            cross_decoded_hand = torch.clamp(cross_decoded_hand, -1.0, 1.0)
            path_c_tips = source_fk_model.forward(cross_decoded_hand)

        n_common = min(source_gt_tips.shape[1], path_c_tips.shape[1])
        src_c_diff = torch.norm(source_gt_tips[:, :n_common] - path_c_tips[:, :n_common], dim=-1) * 1000

        print("\n  {}:".format(hand_name))
        print("    Path A vs B (z→dec vs z'→dec, same hand):     mean={:.2f}mm  max={:.2f}mm".format(
            ab_diff.mean().item(), ab_diff.max().item()))
        print("    Path C: z'→source_dec vs source_gt:           mean={:.2f}mm  max={:.2f}mm".format(
            src_c_diff.mean().item(), src_c_diff.max().item()))

    print("\n" + "=" * 70 + "\n")

    return decoded


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DexLatent visualization (Isaac Lab)")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--hands", nargs="+", default=["mano_right", "shadow_right", "inspire_right"])
    parser.add_argument("--source", type=str, default="mano_right")
    parser.add_argument("--motion_pkl", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--num_keyframes", type=int, default=10)
    parser.add_argument("--spacing", type=float, default=0.4)
    args = parser.parse_args()

    import engines.engine as engine_module
    from engines.engine_builder import build_engine

    engine_config = {
        "engine_name": "isaac_lab",
        "control_mode": "pos",
        "control_freq": 30,
        "sim_freq": 120,
        "env_spacing": 5,
    }

    num_envs = 1
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    eng = build_engine(engine_config, num_envs, device, visualize=True)

    source_key = f"{args.source}_source"
    all_vis = [source_key] + list(args.hands)

    env_id = eng.create_env()

    hand_obj_ids: Dict[str, int] = {}
    hand_y_offsets: Dict[str, float] = {}

    for i, vis_name in enumerate(all_vis):
        hand_for_asset = args.source if vis_name == source_key else vis_name
        config = HAND_CONFIGS[hand_for_asset]
        usd_path = str(config["usd_path"])
        art_cfg = build_articulation_cfg(hand_for_asset)

        y_off = (i - len(all_vis) / 2.0 + 0.5) * args.spacing
        start_pos = np.array([0.0, y_off, 0.3])

        obj_id = eng.create_obj(
            env_id=env_id,
            obj_type=engine_module.ObjType.articulated,
            asset_file=usd_path,
            name=vis_name.replace("_source", "_src"),
            fix_root=False,
            start_pos=start_pos,
            enable_self_collisions=False,
            articulation_cfg=art_cfg,
        )
        hand_obj_ids[vis_name] = obj_id
        hand_y_offsets[vis_name] = y_off

    eng.initialize_sim()

    config = TrainingConfig(device=torch.device("cpu"))
    trainer = CrossEmbodimentTrainer(args.hands, config)

    all_joints_to_engine: Dict[str, torch.Tensor] = {}
    for vis_name in all_vis:
        obj_id = hand_obj_ids[vis_name]
        fk_hand = args.source if vis_name == source_key else vis_name
        fk_model = trainer.hand_models[fk_hand]
        dummy = torch.zeros(1, fk_model.dof_count())
        all_names, _ = fk_model.expand_to_all_joints(dummy)
        reorder = build_all_joints_to_engine_reorder(eng, obj_id, all_names)
        all_joints_to_engine[vis_name] = reorder

    eng.set_camera_pose(
        pos=np.array([1.5, 0.0, 0.8]),
        look_at=np.array([0.0, 0.0, 0.3]),
    )

    trainer.load_checkpoint(args.ckpt)

    if args.motion_pkl is not None:
        trajectories = generate_pkl_trajectory(trainer, args.source, args.motion_pkl)
        num_frames = trajectories[source_key].shape[0]
    else:
        trajectories = generate_smooth_trajectory(
            trainer, args.source, args.num_frames, args.num_keyframes)
        num_frames = args.num_frames

    wrist_dof = trainer.config.wrist_dof

    # ---- Numerical evaluation ----
    print("\n" + "=" * 70)
    print("  NUMERICAL EVALUATION")
    print("=" * 70)

    source_traj = trajectories[source_key]  # [wrist(6), hand(N)]
    T_eval = source_traj.shape[0]

    for hand_name in args.hands:
        decoded_traj = trajectories[hand_name]
        fk_model_src = trainer.hand_models[args.source]
        fk_model_tgt = trainer.hand_models[hand_name]

        # Compute fingertip positions using training FK (with alignment + wrist)
        from dexlatent.kinematics import wrist_dof_to_transform
        src_hand = source_traj[:, wrist_dof:]
        tgt_hand = decoded_traj[:, wrist_dof:]
        src_wrist = source_traj[:, :wrist_dof]
        tgt_wrist = decoded_traj[:, :wrist_dof]

        with torch.no_grad():
            src_wrist_T = wrist_dof_to_transform(src_wrist)
            tgt_wrist_T = wrist_dof_to_transform(tgt_wrist)
            src_tips = fk_model_src.forward(src_hand, wrist_transform=src_wrist_T)  # (T, 5, 3)
            tgt_tips = fk_model_tgt.forward(tgt_hand, wrist_transform=tgt_wrist_T)  # (T, 5, 3)

        # Per-finger tip distance (handle different tip counts)
        all_tip_names = ["thumb", "index", "middle", "ring", "pinky"]
        n_src = src_tips.shape[1]
        n_tgt = tgt_tips.shape[1]
        n_common = min(n_src, n_tgt)
        tip_names = all_tip_names[:n_common]
        tip_dists = torch.linalg.norm(src_tips[:, :n_common] - tgt_tips[:, :n_common], dim=-1)

        # Inter-finger distances (pinch pairs: thumb-index)
        src_thumb_idx_dist = torch.linalg.norm(src_tips[:, 0] - src_tips[:, 1], dim=-1)
        tgt_thumb_idx_dist = torch.linalg.norm(tgt_tips[:, 0] - tgt_tips[:, 1], dim=-1)
        pinch_dist_err = (src_thumb_idx_dist - tgt_thumb_idx_dist).abs()

        # Hand reconstruction (normalized space)
        if hand_name == args.source:
            hand_rec_err = (src_hand - tgt_hand).abs().mean().item()
        else:
            hand_rec_err = None

        print(f"\n  [{args.source} -> {hand_name}] ({T_eval} frames, {n_src} vs {n_tgt} tips, comparing {n_common})")
        print(f"  {'finger':<10s} {'mean_dist(mm)':<15s} {'max_dist(mm)':<15s} {'std(mm)':<10s}")
        print(f"  {'-'*50}")
        for fi, fname in enumerate(tip_names):
            d = tip_dists[:, fi] * 1000
            print(f"  {fname:<10s} {d.mean().item():<15.2f} {d.max().item():<15.2f} {d.std().item():<10.2f}")
        overall = tip_dists.mean(dim=1) * 1000
        print(f"  {'OVERALL':<10s} {overall.mean().item():<15.2f} {overall.max().item():<15.2f} {overall.std().item():<10.2f}")

        print(f"\n  Thumb-Index pinch distance error: {pinch_dist_err.mean().item()*1000:.2f} mm (mean), {pinch_dist_err.max().item()*1000:.2f} mm (max)")

        if hand_rec_err is not None:
            print(f"  Self-reconstruction error (normalized qpos): {hand_rec_err:.6f}")

    print("")

    frame_idx = 0
    while True:
        t = frame_idx % num_frames

        for vis_name in all_vis:
            obj_id = hand_obj_ids[vis_name]
            reorder = all_joints_to_engine[vis_name]
            fk_hand = args.source if vis_name == source_key else vis_name
            fk_model = trainer.hand_models[fk_hand]

            full_qpos = trajectories[vis_name][t]
            wrist_part = full_qpos[:wrist_dof]
            hand_part = full_qpos[wrist_dof:]

            all_names, all_angles = fk_model.expand_to_all_joints(hand_part.unsqueeze(0))
            all_angles_reordered = all_angles[:, reorder]
            eng.set_cmd(obj_id, all_angles_reordered.to(device=device))
            eng.set_dof_pos(None, obj_id, all_angles_reordered.to(device=device))

            root_pos = wrist_part[:3].clone()
            root_pos[1] += hand_y_offsets[vis_name]
            eng.set_root_pos(None, obj_id, root_pos.to(device=device).unsqueeze(0))

            wrist_rot_aa = wrist_part[3:6].numpy()
            wrist_rot_mat = ScipyRotation.from_rotvec(wrist_rot_aa).as_matrix()

            src_align = ALIGNMENT_ROTATIONS.get(args.source, torch.eye(3)).numpy()
            tgt_align = ALIGNMENT_ROTATIONS.get(fk_hand, torch.eye(3)).numpy()
            relative_align = src_align.T @ tgt_align
            combined_rot = wrist_rot_mat @ relative_align

            quat_xyzw = ScipyRotation.from_matrix(combined_rot).as_quat()
            quat_tensor = torch.tensor(quat_xyzw, dtype=torch.float32, device=device).unsqueeze(0)
            eng.set_root_rot(None, obj_id, quat_tensor)

            eng.set_root_vel(None, obj_id, torch.zeros(1, 3, device=device))
            eng.set_root_ang_vel(None, obj_id, torch.zeros(1, 3, device=device))

        eng.step()
        eng.render()
        frame_idx += 1


if __name__ == "__main__":
    main()
