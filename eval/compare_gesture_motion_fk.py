#!/usr/bin/env python3
"""Compare fingertip and wrist movement using URDF forward kinematics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
MANO_DOF_NAMES = (
    "j_index1y", "j_index1z", "j_index2", "j_index3",
    "j_middle1y", "j_middle1z", "j_middle2", "j_middle3",
    "j_pinky1y", "j_pinky1z", "j_pinky2", "j_pinky3",
    "j_ring1y", "j_ring1z", "j_ring2", "j_ring3",
    "j_thumb1x", "j_thumb1y", "j_thumb1z",
    "j_thumb2y", "j_thumb2z", "j_thumb3",
)


@dataclass(frozen=True)
class HandConfig:
    urdf_name: str
    root_link: str
    tip_links: dict[str, str]


HAND_CONFIGS = {
    "mano_right": HandConfig(
        "rh_mano.urdf",
        "palm",
        {
            "thumb": "thumb_tip",
            "index": "index_tip",
            "middle": "middle_tip",
            "ring": "ring_tip",
            "pinky": "pinky_tip",
        },
    ),
    "shadow_right": HandConfig(
        "shadow_hand_right.urdf",
        "palm",
        {
            "thumb": "thtip",
            "index": "fftip",
            "middle": "mftip",
            "ring": "rftip",
            "pinky": "lftip",
        },
    ),
    "inspire_right": HandConfig(
        "inspire_hand_right.urdf",
        "R_hand_base_link",
        {
            "thumb": "R_thumb_tip",
            "index": "R_index_tip",
            "middle": "R_middle_tip",
            "ring": "R_ring_tip",
            "pinky": "R_pinky_tip",
        },
    ),
    "allegro_right": HandConfig(
        "allegro_hand_right.urdf",
        "base_link",
        {
            "thumb": "link_15.0_tip",
            "index": "link_3.0_tip",
            "middle": "link_7.0_tip",
            "ring": "link_11.0_tip",
        },
    ),
}


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    mimic_joint: str | None
    mimic_multiplier: float
    mimic_offset: float


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "result" / "mouse" / "mouse"
    parser = argparse.ArgumentParser(
        description="Compare hand trajectories using URDF forward kinematics."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=data_dir / "102_sv_dict.pkl",
    )
    parser.add_argument(
        "--source-hand",
        choices=HAND_CONFIGS,
        default="mano_right",
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
    )
    parser.add_argument(
        "--urdf-dir",
        type=Path,
        default=root / "result" / "urdf",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "eval" / "gesture_motion_fk_results",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Odd moving-average window for speed profiles (default: 5).",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def load_pickle(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"PKL not found: {path}")
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict) or "opt_dof_pos" not in data:
        raise ValueError(f"{path} must contain an opt_dof_pos array")
    return data


def vector(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    values = np.fromstring(text, sep=" ", dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"Expected three values, got: {text!r}")
    return values


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


class UrdfKinematics:
    def __init__(self, path: Path, root_link: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"URDF not found: {path}")
        xml_root = ET.parse(path).getroot()
        self.root_link = root_link
        self.joints: list[Joint] = []
        self.children: dict[str, list[Joint]] = {}
        for element in xml_root.findall("joint"):
            parent_element = element.find("parent")
            child_element = element.find("child")
            if parent_element is None or child_element is None:
                continue
            origin_element = element.find("origin")
            xyz = vector(
                origin_element.get("xyz") if origin_element is not None else None,
                (0.0, 0.0, 0.0),
            )
            rpy = vector(
                origin_element.get("rpy") if origin_element is not None else None,
                (0.0, 0.0, 0.0),
            )
            axis_element = element.find("axis")
            axis = vector(
                axis_element.get("xyz") if axis_element is not None else None,
                (1.0, 0.0, 0.0),
            )
            mimic_element = element.find("mimic")
            joint = Joint(
                name=element.get("name", ""),
                kind=element.get("type", "fixed"),
                parent=parent_element.get("link", ""),
                child=child_element.get("link", ""),
                origin=transform(rpy_matrix(rpy), xyz),
                axis=axis,
                mimic_joint=(
                    mimic_element.get("joint") if mimic_element is not None else None
                ),
                mimic_multiplier=float(
                    mimic_element.get("multiplier", "1.0")
                    if mimic_element is not None else 1.0
                ),
                mimic_offset=float(
                    mimic_element.get("offset", "0.0")
                    if mimic_element is not None else 0.0
                ),
            )
            self.joints.append(joint)
            self.children.setdefault(joint.parent, []).append(joint)

        self.traversal: list[Joint] = []
        queue = [root_link]
        visited = {root_link}
        while queue:
            parent = queue.pop(0)
            for joint in self.children.get(parent, []):
                self.traversal.append(joint)
                if joint.child not in visited:
                    visited.add(joint.child)
                    queue.append(joint.child)

    def forward(
        self,
        angles: dict[str, float],
        requested_links: list[str],
    ) -> dict[str, np.ndarray]:
        transforms = {self.root_link: np.eye(4, dtype=np.float64)}
        resolved_angles = dict(angles)
        for joint in self.traversal:
            parent = transforms[joint.parent]
            joint_transform = np.eye(4, dtype=np.float64)
            angle = resolved_angles.get(joint.name)
            if angle is None and joint.mimic_joint is not None:
                parent_angle = resolved_angles.get(joint.mimic_joint, 0.0)
                angle = parent_angle * joint.mimic_multiplier + joint.mimic_offset
                resolved_angles[joint.name] = angle
            if angle is None:
                angle = 0.0
            if joint.kind in {"revolute", "continuous"}:
                joint_transform[:3, :3] = axis_angle_matrix(joint.axis, angle)
            elif joint.kind == "prismatic":
                joint_transform[:3, 3] = joint.axis * angle
            transforms[joint.child] = parent @ joint.origin @ joint_transform
        missing = [link for link in requested_links if link not in transforms]
        if missing:
            raise ValueError(
                f"Links are not reachable from root {self.root_link!r}: {missing}"
            )
        return {link: transforms[link] for link in requested_links}


def canonical_joint_name(name: str) -> str:
    value = name.strip()
    if value.startswith(("r_", "l_")):
        value = value[2:]
    return value.replace(".0", "")


def map_joint_name(hand_name: str, pkl_name: str) -> str:
    value = canonical_joint_name(pkl_name)
    if hand_name == "shadow_right":
        match = re.fullmatch(r"(FFJ|MFJ|RFJ|LFJ|THJ)(\d+)", value)
        if match:
            return f"{match.group(1)}{int(match.group(2)) + 1}"
    if hand_name == "allegro_right":
        match = re.fullmatch(r"joint_(\d+)", value)
        if match:
            return f"joint_{match.group(1)}.0"
    return value


def get_joint_names(data: dict[str, Any], path: Path) -> list[str]:
    qpos = np.asarray(data["opt_dof_pos"])
    names = data.get("meta_joint_names")
    if names is None and qpos.shape[1] == len(MANO_DOF_NAMES):
        names = MANO_DOF_NAMES
    if names is None or len(names) != qpos.shape[1]:
        raise ValueError(f"Cannot determine joint names for {path}")
    return [str(name) for name in names]


def infer_hand_name(data: dict[str, Any], path: Path) -> str:
    hand_name = data.get("meta_target_hand")
    if hand_name in HAND_CONFIGS:
        return str(hand_name)
    for candidate in HAND_CONFIGS:
        if candidate in path.stem:
            return candidate
    raise ValueError(f"Cannot infer hand type for {path}")


def compute_tip_trajectory(
    data: dict[str, Any],
    path: Path,
    hand_name: str,
    urdf_dir: Path,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    config = HAND_CONFIGS[hand_name]
    model = UrdfKinematics(urdf_dir / config.urdf_name, config.root_link)
    qpos = np.asarray(data["opt_dof_pos"], dtype=np.float64)
    joint_names = get_joint_names(data, path)
    mapped_names = [map_joint_name(hand_name, name) for name in joint_names]
    urdf_joint_names = {joint.name for joint in model.joints}
    missing_joints = [
        name for name in mapped_names if name not in urdf_joint_names
    ]
    if missing_joints:
        raise ValueError(
            f"{path}: mapped joints absent from {config.urdf_name}: {missing_joints}"
        )

    fingers = [finger for finger in FINGERS if finger in config.tip_links]
    links = [config.tip_links[finger] for finger in fingers]
    tips = np.empty((qpos.shape[0], len(fingers), 3), dtype=np.float64)
    for frame, row in enumerate(qpos):
        angles = dict(zip(mapped_names, row))
        poses = model.forward(angles, links)
        tips[frame] = np.stack([poses[link][:3, 3] for link in links])

    palm_to_tip = np.linalg.norm(tips, axis=-1)
    scale = float(np.median(palm_to_tip))
    if not np.isfinite(scale) or scale < 1e-6:
        raise ValueError(f"{path}: invalid FK hand scale {scale}")
    metadata = {
        "hand_name": hand_name,
        "urdf": str((urdf_dir / config.urdf_name).resolve()),
        "mapped_joint_names": mapped_names,
        "hand_scale_m": scale,
    }
    return tips, fingers, metadata


def resample(values: np.ndarray, length: int) -> np.ndarray:
    if values.shape[0] == length:
        return values
    old_time = np.linspace(0.0, 1.0, values.shape[0])
    new_time = np.linspace(0.0, 1.0, length)
    flat = values.reshape(values.shape[0], -1)
    sampled = np.column_stack(
        [np.interp(new_time, old_time, flat[:, index]) for index in range(flat.shape[1])]
    )
    return sampled.reshape((length,) + values.shape[1:])


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("--smooth-window must be a positive odd integer")
    if window == 1:
        return values.copy()
    actual = min(window, values.shape[0] if values.shape[0] % 2 else values.shape[0] - 1)
    if actual <= 1:
        return values.copy()
    radius = actual // 2
    kernel = np.ones(actual, dtype=np.float64) / actual
    flat = values.reshape(values.shape[0], -1)
    output = np.empty_like(flat)
    for index in range(flat.shape[1]):
        padded = np.pad(flat[:, index], (radius, radius), mode="edge")
        output[:, index] = np.convolve(padded, kernel, mode="valid")
    return output.reshape(values.shape)


def normalized_rmse(reference: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.sqrt(np.mean(reference ** 2)))
    error = float(np.sqrt(np.mean((reference - target) ** 2)))
    return error / max(denominator, 1e-8)


def correlation_error(reference: np.ndarray, target: np.ndarray) -> float:
    reference = reference.reshape(-1) - np.mean(reference)
    target = target.reshape(-1) - np.mean(target)
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(target))
    if denominator < 1e-12:
        return 0.0 if np.allclose(reference, target) else 0.5
    correlation = float(np.dot(reference, target) / denominator)
    return 0.5 * (1.0 - float(np.clip(correlation, 0.0, 1.0)))


def speed_features(tips: np.ndarray, scale: float, window: int) -> np.ndarray:
    speeds = np.linalg.norm(np.diff(tips, axis=0), axis=-1) / scale
    speeds = smooth(speeds, window)
    mean_speed = np.mean(speeds, axis=0, keepdims=True)
    return speeds / np.maximum(mean_speed, 1e-8)


def wrist_speed_features(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(data["opt_wrist_pos"], dtype=np.float64)
    rotation = np.asarray(data["opt_wrist_rot"], dtype=np.float64)
    position_speed = np.linalg.norm(np.diff(position, axis=0), axis=-1)
    rotation_speed = np.linalg.norm(np.diff(rotation, axis=0), axis=-1)
    position_speed /= max(float(np.mean(position_speed)), 1e-8)
    rotation_speed /= max(float(np.mean(rotation_speed)), 1e-8)
    return position_speed, rotation_speed


def compare_wrist(source: dict[str, Any], target: dict[str, Any]) -> float:
    source_position, source_rotation = wrist_speed_features(source)
    target_position, target_rotation = wrist_speed_features(target)
    length = max(len(source_position), len(target_position))
    source_position = resample(source_position[:, None], length)[:, 0]
    target_position = resample(target_position[:, None], length)[:, 0]
    source_rotation = resample(source_rotation[:, None], length)[:, 0]
    target_rotation = resample(target_rotation[:, None], length)[:, 0]
    return 0.5 * (
        normalized_rmse(source_position, target_position)
        + normalized_rmse(source_rotation, target_rotation)
    )


def compare_target(
    source_data: dict[str, Any],
    source_tips: np.ndarray,
    source_fingers: list[str],
    source_scale: float,
    target_data: dict[str, Any],
    target_tips: np.ndarray,
    target_fingers: list[str],
    target_scale: float,
    smooth_window: int,
) -> dict[str, Any]:
    common = [
        finger for finger in FINGERS
        if finger in source_fingers and finger in target_fingers
    ]
    source_indices = [source_fingers.index(finger) for finger in common]
    target_indices = [target_fingers.index(finger) for finger in common]
    source = source_tips[:, source_indices]
    target = target_tips[:, target_indices]
    frame_count = max(source.shape[0], target.shape[0])
    source = resample(source, frame_count)
    target = resample(target, frame_count)

    source_speed = speed_features(source, source_scale, smooth_window)
    target_speed = speed_features(target, target_scale, smooth_window)
    movement_error = correlation_error(source_speed, target_speed)
    wrist_error = compare_wrist(source_data, target_data)

    overall_error = 0.80 * movement_error + 0.20 * wrist_error
    return {
        "common_fingers": common,
        "excluded_fingers": [
            finger for finger in FINGERS if finger not in target_fingers
        ],
        "movement_error": movement_error,
        "wrist_error": wrist_error,
        "overall_error": overall_error,
        "similarity_index": 100.0 * math.exp(-overall_error),
    }


def write_csv(results: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "target", "overall_error", "similarity_index", "movement_error",
        "wrist_error", "common_fingers",
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
        "Tip movement": [result["movement_error"] for result in results],
        "Wrist movement": [result["wrist_error"] for result in results],
        "Overall": [result["overall_error"] for result in results],
    }
    x = np.arange(len(labels))
    width = 0.24
    figure, axis = plt.subplots(figsize=(11, 5.5))
    for offset, (name, values) in enumerate(metrics.items()):
        axis.bar(x + (offset - 1) * width, values, width, label=name)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Normalized error (lower is better)")
    axis.set_title("URDF/FK fingertip and wrist movement comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "fk_summary_errors.png", dpi=180)
    plt.close(figure)


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\nURDF/FK fingertip comparison (lower error is better)\n")
    header = (
        f"{'target':<18} {'overall':>8} {'sim.index':>10} "
        f"{'movement':>9} {'wrist':>8}"
    )
    print(header)
    print("-" * len(header))
    for result in sorted(results, key=lambda item: item["overall_error"]):
        print(
            f"{result['target']:<18} {result['overall_error']:>8.4f} "
            f"{result['similarity_index']:>10.2f} "
            f"{result['movement_error']:>9.4f} {result['wrist_error']:>8.4f}"
        )
        if result["excluded_fingers"]:
            print("  excluded unavailable fingers: " + ", ".join(result["excluded_fingers"]))


def main() -> None:
    args = parse_args()
    source_data = load_pickle(args.source)
    source_tips, source_fingers, source_meta = compute_tip_trajectory(
        source_data, args.source, args.source_hand, args.urdf_dir
    )
    results = []
    for target_path in args.targets:
        target_data = load_pickle(target_path)
        target_hand = infer_hand_name(target_data, target_path)
        target_tips, target_fingers, target_meta = compute_tip_trajectory(
            target_data, target_path, target_hand, args.urdf_dir
        )
        result = compare_target(
            source_data,
            source_tips,
            source_fingers,
            source_meta["hand_scale_m"],
            target_data,
            target_tips,
            target_fingers,
            target_meta["hand_scale_m"],
            args.smooth_window,
        )
        result.update(
            {
                "target": target_hand,
                "path": str(target_path.resolve()),
                "frames": int(target_tips.shape[0]),
                "fk_metadata": target_meta,
            }
        )
        results.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(args.source.resolve()),
        "source_fk_metadata": source_meta,
        "method": {
            "movement": "correlation distance of normalized fingertip speeds",
            "wrist": "RMSE of normalized wrist translation/rotation speeds",
            "overall_weights": {
                "movement": 0.80,
                "wrist": 0.20,
            },
        },
        "results": results,
    }
    with (args.output_dir / "comparison_fk.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    write_csv(results, args.output_dir / "comparison_fk.csv")
    if not args.no_plots:
        write_plots(results, args.output_dir)
    print_summary(results)
    print(f"\nSaved new FK results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
