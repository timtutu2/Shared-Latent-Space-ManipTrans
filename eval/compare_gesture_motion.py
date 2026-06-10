#!/usr/bin/env python3
"""Compare cross-embodiment hand trajectories without using world coordinates.

The retargeted PKLs contain joint angles but not Cartesian joint positions, so
direct fingertip distances cannot be computed without each robot's URDF. Raw
joint-vector errors are also misleading because each robot uses a different
mechanism to create the same visible pose. This script instead compares:

1. Gesture: a normalized per-finger activity/progression curve that measures
   how far the finger has moved from its initial configuration.
2. Movement: the smoothed rate of change of that progression curve.
3. Wrist movement: normalized translation and rotation-speed profiles.

The finger errors are correlation distances and are therefore insensitive to
joint ranges, joint count, and how motion is distributed among coupled joints.
An error of 0 is ideal. The similarity index is 100 * exp(-overall_error).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np


FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# The original MANO/ArtiMano pickle predates the metadata fields written to the
# decoded files. This is the order used by ManipTrans_Lab's Artimano hand.
MANO_DOF_NAMES = (
    "j_index1y", "j_index1z", "j_index2", "j_index3",
    "j_middle1y", "j_middle1z", "j_middle2", "j_middle3",
    "j_pinky1y", "j_pinky1z", "j_pinky2", "j_pinky3",
    "j_ring1y", "j_ring1z", "j_ring2", "j_ring3",
    "j_thumb1x", "j_thumb1y", "j_thumb1z",
    "j_thumb2y", "j_thumb2z", "j_thumb3",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "result" / "mouse" / "mouse"
    parser = argparse.ArgumentParser(
        description="Compare gesture and motion across retargeted robot hands."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=data_dir / "102_sv_dict.pkl",
        help="Reference MANO/ArtiMano PKL.",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        nargs="+",
        default=[
            data_dir / "102_sv_dict_shadow_right.pkl",
            data_dir / "102_sv_dict_allegro_right.pkl",
            data_dir / "102_sv_dict_inspire_right.pkl",
        ],
        help="One or more retargeted robot-hand PKLs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "eval" / "gesture_motion_results",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG generation (JSON and CSV are still written).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=9,
        help="Odd moving-average window for finger progression (default: 9).",
    )
    return parser.parse_args()


def load_pickle(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"PKL not found: {path}")
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dictionary in {path}, got {type(data).__name__}")
    if "opt_dof_pos" not in data:
        raise KeyError(f"{path} does not contain 'opt_dof_pos'")
    return data


def hand_label(path: Path, data: dict[str, Any]) -> str:
    label = data.get("meta_target_hand")
    if isinstance(label, str):
        return label
    return path.stem


def get_dof_names(data: dict[str, Any], path: Path) -> list[str]:
    qpos = np.asarray(data["opt_dof_pos"])
    names = data.get("meta_joint_names")
    if names is not None:
        names = [str(name) for name in names]
    elif qpos.shape[1] == len(MANO_DOF_NAMES):
        names = list(MANO_DOF_NAMES)
    else:
        raise ValueError(
            f"{path} has {qpos.shape[1]} DOFs and no meta_joint_names. "
            "Add its joint names to the PKL or this script."
        )
    if len(names) != qpos.shape[1]:
        raise ValueError(
            f"{path}: {len(names)} joint names for {qpos.shape[1]} DOF columns"
        )
    return names


def canonical_name(name: str) -> str:
    value = name.lower().replace(".", "_")
    for prefix in ("r_", "l_"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    return value


def finger_for_joint(name: str) -> str | None:
    name = canonical_name(name)
    allegro_match = re.fullmatch(r"joint_(\d+)", name)
    if allegro_match:
        joint_number = int(allegro_match.group(1))
        if 0 <= joint_number <= 3:
            return "index"
        if 4 <= joint_number <= 7:
            return "middle"
        if 8 <= joint_number <= 11:
            return "ring"
        if 12 <= joint_number <= 15:
            return "thumb"
    aliases = {
        "thumb": ("thumb", "thj"),
        "index": ("index", "ffj"),
        "middle": ("middle", "mfj"),
        "ring": ("ring", "rfj"),
        "pinky": ("pinky", "little", "lfj"),
    }
    for finger, tokens in aliases.items():
        if any(token in name for token in tokens):
            return finger
    return None


def split_fingers(data: dict[str, Any], path: Path) -> dict[str, np.ndarray]:
    qpos = np.asarray(data["opt_dof_pos"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[0] < 2:
        raise ValueError(f"{path}: opt_dof_pos must have shape [T, D], T >= 2")
    names = get_dof_names(data, path)
    groups: dict[str, list[int]] = {finger: [] for finger in FINGERS}
    unknown = []
    for index, name in enumerate(names):
        finger = finger_for_joint(name)
        if finger is None:
            unknown.append(name)
        else:
            groups[finger].append(index)
    if unknown:
        raise ValueError(f"{path}: could not assign joints to fingers: {unknown}")
    return {
        finger: qpos[:, indices]
        for finger, indices in groups.items()
        if indices
    }


def resample(values: np.ndarray, length: int) -> np.ndarray:
    if values.shape[0] == length:
        return values
    old_time = np.linspace(0.0, 1.0, values.shape[0])
    new_time = np.linspace(0.0, 1.0, length)
    flat = values.reshape(values.shape[0], -1)
    sampled = np.column_stack(
        [np.interp(new_time, old_time, flat[:, column]) for column in range(flat.shape[1])]
    )
    return sampled.reshape((length,) + values.shape[1:])


def smooth_curve(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("--smooth-window must be a positive odd integer")
    if window == 1:
        return values.copy()
    window = min(window, values.shape[0] if values.shape[0] % 2 else values.shape[0] - 1)
    if window <= 1:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def finger_progression(qpos: np.ndarray, smooth_window: int) -> np.ndarray:
    """Return mechanism-invariant distance from the initial finger pose."""
    joint_ranges = np.ptp(qpos, axis=0)
    active = joint_ranges > 1e-5
    if not np.any(active):
        return np.zeros(qpos.shape[0], dtype=np.float64)
    normalized_delta = (
        qpos[:, active] - qpos[0:1, active]
    ) / joint_ranges[active]
    progression = np.linalg.norm(normalized_delta, axis=1) / math.sqrt(active.sum())
    return smooth_curve(progression, smooth_window)


def correlation_error(reference: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Map positive correlation to [0, 0.5] error; 1 correlation gives 0."""
    reference = reference - np.mean(reference)
    target = target - np.mean(target)
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(target))
    if denominator < 1e-12:
        correlation = 1.0 if np.allclose(reference, target) else 0.0
    else:
        correlation = float(np.dot(reference, target) / denominator)
    correlation = float(np.clip(correlation, 0.0, 1.0))
    return 0.5 * (1.0 - correlation), correlation


def compare_finger(
    source: np.ndarray,
    target: np.ndarray,
    smooth_window: int,
) -> dict[str, float]:
    frame_count = max(source.shape[0], target.shape[0])
    source = resample(source, frame_count)
    target = resample(target, frame_count)
    source_progression = finger_progression(source, smooth_window)
    target_progression = finger_progression(target, smooth_window)
    gesture_error, gesture_correlation = correlation_error(
        source_progression, target_progression
    )
    source_movement = smooth_curve(
        np.abs(np.gradient(source_progression)), smooth_window
    )
    target_movement = smooth_curve(
        np.abs(np.gradient(target_progression)), smooth_window
    )
    movement_error, movement_correlation = correlation_error(
        source_movement, target_movement
    )
    return {
        "gesture_error": gesture_error,
        "gesture_correlation": gesture_correlation,
        "movement_error": movement_error,
        "movement_correlation": movement_correlation,
        "source_dofs": int(source.shape[1]),
        "target_dofs": int(target.shape[1]),
    }


def normalized_speed(values: np.ndarray) -> np.ndarray:
    speed = np.linalg.norm(np.diff(values, axis=0), axis=-1)
    scale = float(np.mean(speed))
    if scale < 1e-12:
        return np.zeros_like(speed)
    return speed / scale


def relative_rmse(reference: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.sqrt(np.mean(np.square(reference))))
    if denominator < 1e-12:
        return float(np.sqrt(np.mean(np.square(target))))
    return float(np.sqrt(np.mean(np.square(reference - target))) / denominator)


def wrist_speed_features(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(data.get("opt_wrist_pos"), dtype=np.float64)
    rotation = np.asarray(data.get("opt_wrist_rot"), dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("opt_wrist_pos must have shape [T, 3]")
    if rotation.ndim != 2 or rotation.shape[1] != 3:
        raise ValueError("opt_wrist_rot must have shape [T, 3]")
    return normalized_speed(position), normalized_speed(rotation)


def compare_wrist(source: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    source_position, source_rotation = wrist_speed_features(source)
    target_position, target_rotation = wrist_speed_features(target)
    length = max(len(source_position), len(target_position))
    source_position = resample(source_position[:, None], length)[:, 0]
    target_position = resample(target_position[:, None], length)[:, 0]
    source_rotation = resample(source_rotation[:, None], length)[:, 0]
    target_rotation = resample(target_rotation[:, None], length)[:, 0]
    return {
        "translation_movement_error": relative_rmse(source_position, target_position),
        "rotation_movement_error": relative_rmse(source_rotation, target_rotation),
    }


def compare_target(
    source_data: dict[str, Any],
    source_path: Path,
    target_data: dict[str, Any],
    target_path: Path,
    smooth_window: int,
) -> dict[str, Any]:
    source_fingers = split_fingers(source_data, source_path)
    target_fingers = split_fingers(target_data, target_path)
    common_fingers = [
        finger for finger in FINGERS
        if finger in source_fingers and finger in target_fingers
    ]
    if not common_fingers:
        raise ValueError(f"No common fingers found between {source_path} and {target_path}")

    finger_results = {
        finger: compare_finger(
            source_fingers[finger], target_fingers[finger], smooth_window
        )
        for finger in common_fingers
    }
    gesture_error = float(np.mean([
        result["gesture_error"] for result in finger_results.values()
    ]))
    finger_movement_error = float(np.mean([
        result["movement_error"] for result in finger_results.values()
    ]))
    wrist = compare_wrist(source_data, target_data)
    wrist_movement_error = float(np.mean(list(wrist.values())))

    # Gesture gets the largest weight because the user's main question is
    # whether the hand shape evolves in the same way.
    overall_error = (
        0.60 * gesture_error
        + 0.30 * finger_movement_error
        + 0.10 * wrist_movement_error
    )
    return {
        "target": hand_label(target_path, target_data),
        "path": str(target_path.resolve()),
        "frames": int(np.asarray(target_data["opt_dof_pos"]).shape[0]),
        "common_fingers": common_fingers,
        "missing_source_fingers": [
            finger for finger in FINGERS if finger not in target_fingers
        ],
        "finger_results": finger_results,
        "gesture_error": gesture_error,
        "finger_movement_error": finger_movement_error,
        "wrist_movement_error": wrist_movement_error,
        "wrist_results": wrist,
        "overall_error": overall_error,
        "similarity_index": 100.0 * math.exp(-overall_error),
    }


def write_csv(results: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "target", "overall_error", "similarity_index", "gesture_error",
        "finger_movement_error", "wrist_movement_error", "common_fingers",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {field: result[field] for field in fields}
            row["common_fingers"] = " ".join(result["common_fingers"])
            writer.writerow(row)


def write_plots(results: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping plots.")
        return

    labels = [result["target"] for result in results]
    metrics = {
        "Gesture": [result["gesture_error"] for result in results],
        "Finger movement": [result["finger_movement_error"] for result in results],
        "Wrist movement": [result["wrist_movement_error"] for result in results],
        "Overall": [result["overall_error"] for result in results],
    }
    x = np.arange(len(labels))
    width = 0.19
    figure, axis = plt.subplots(figsize=(10, 5))
    for offset, (name, values) in enumerate(metrics.items()):
        axis.bar(x + (offset - 1.5) * width, values, width, label=name)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Normalized error (lower is better)")
    axis.set_title("Cross-embodiment gesture and movement comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "summary_errors.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(
        len(results), 1, figsize=(10, max(3.0, 2.7 * len(results))), squeeze=False
    )
    for axis, result in zip(axes[:, 0], results):
        fingers = result["common_fingers"]
        gesture = [
            result["finger_results"][finger]["gesture_error"] for finger in fingers
        ]
        movement = [
            result["finger_results"][finger]["movement_error"] for finger in fingers
        ]
        positions = np.arange(len(fingers))
        axis.bar(positions - 0.18, gesture, 0.36, label="Gesture")
        axis.bar(positions + 0.18, movement, 0.36, label="Movement")
        axis.set_xticks(positions, fingers)
        axis.set_ylabel("Error")
        axis.set_title(result["target"])
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "per_finger_errors.png", dpi=180)
    plt.close(figure)


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\nGesture and movement comparison (lower error is better)\n")
    header = (
        f"{'target':<18} {'overall':>9} {'sim. index':>11} "
        f"{'gesture':>9} {'finger mov.':>12} {'wrist mov.':>11}"
    )
    print(header)
    print("-" * len(header))
    for result in sorted(results, key=lambda item: item["overall_error"]):
        print(
            f"{result['target']:<18} "
            f"{result['overall_error']:>9.4f} "
            f"{result['similarity_index']:>10.2f} "
            f"{result['gesture_error']:>9.4f} "
            f"{result['finger_movement_error']:>12.4f} "
            f"{result['wrist_movement_error']:>11.4f}"
        )
        if result["missing_source_fingers"]:
            missing = ", ".join(result["missing_source_fingers"])
            print(f"  excluded unavailable fingers: {missing}")


def main() -> None:
    args = parse_args()
    source_data = load_pickle(args.source)
    results = []
    for target_path in args.targets:
        target_data = load_pickle(target_path)
        results.append(
            compare_target(
                source_data,
                args.source,
                target_data,
                target_path,
                args.smooth_window,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(args.source.resolve()),
        "method": {
            "gesture": "correlation distance of normalized finger progression",
            "finger_movement": "correlation distance of smoothed progression rate",
            "wrist_movement": "normalized translation/rotation speed-profile RMSE",
            "smooth_window": args.smooth_window,
            "overall_weights": {
                "gesture": 0.60,
                "finger_movement": 0.30,
                "wrist_movement": 0.10,
            },
        },
        "results": results,
    }
    with (args.output_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    write_csv(results, args.output_dir / "comparison.csv")
    if not args.no_plots:
        write_plots(results, args.output_dir)
    print_summary(results)
    print(f"\nSaved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
