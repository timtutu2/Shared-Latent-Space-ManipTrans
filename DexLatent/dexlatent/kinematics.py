"""Differentiable forward kinematics for standalone hand embodiments.

Supports Shadow, Inspire, and MANO hands from the ManipTransDexLatent assets.
All revolute joints (including mimic) are treated as independent DOFs to match
the Isaac Lab simulation behavior.

Each hand has a static alignment rotation so all hands face the same canonical
direction (fingers → +x, thumb → -z side). A 6-DOF wrist transform (3 pos +
3 axis-angle) can be applied on top to position the hand in world space.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as et

import torch
import torch.nn.functional as F
from urdf_parser_py import urdf as urdf_parser

# PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# ASSETS_DIR: str = os.path.join(PROJECT_ROOT, "data", "assets")

PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ASSETS_DIR: str = os.path.join(PROJECT_ROOT, "data", "assets")

# Alignment rotations: rotate each hand's local frame so fingers point +x.
# shadow_right:  fingers +z → +x  ⇒  Ry(+90°)
# inspire_right: fingers -y → +x  ⇒  Rz(+90°)
# mano_right:    fingers -x → +x  ⇒  Ry(180°)
ALIGNMENT_ROTATIONS: Dict[str, torch.Tensor] = {
    "shadow_right": torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=torch.float32),   # Ry(+90)
    "shadow_left":  torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=torch.float32),    # Ry(-90)
    "inspire_right": torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32),   # Rz(+90)
    "mano_right": torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32),     # Ry(180)
    "mano_left":  torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32),     # Ry(180)
    "allegro_right": torch.tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=torch.float32),  # Rzn90 @ Rxn90
}

HAND_CONFIGS: Dict[str, Dict[str, object]] = {
    "shadow_right": {
        "urdf_path": os.path.join(ASSETS_DIR, "shadow_hand", "shadow_hand_woarm_right.urdf"),
        "usd_path": os.path.join(ASSETS_DIR, "shadow_hand", "shadow_hand_woarm_right", "shadow_hand_woarm_right.usd"),
        "root_link": "r_palm",
        "tip_links": ("r_thtip", "r_fftip", "r_mftip", "r_rftip", "r_lftip"),
    },
    "shadow_left": {
        "urdf_path": os.path.join(ASSETS_DIR, "shadow_hand", "shadow_hand_woarm_left.urdf"),
        "usd_path": os.path.join(ASSETS_DIR, "shadow_hand", "shadow_hand_woarm_left", "shadow_hand_woarm_left.usd"),
        "root_link": "l_palm",
        "tip_links": ("l_thtip", "l_fftip", "l_mftip", "l_rftip", "l_lftip"),
    },
    "inspire_right": {
        "urdf_path": os.path.join(ASSETS_DIR, "inspire_hand", "inspire_hand_right.urdf"),
        "usd_path": os.path.join(ASSETS_DIR, "inspire_hand", "inspire_hand_right", "inspire_hand_right.usd"),
        "root_link": "R_hand_base_link",
        "tip_links": ("R_thumb_tip", "R_index_tip", "R_middle_tip", "R_ring_tip", "R_pinky_tip"),
    },
    "mano_right": {
        "urdf_path": os.path.join(ASSETS_DIR, "mano", "rh_mano.urdf"),
        "usd_path": os.path.join(ASSETS_DIR, "mano", "rh_mano", "rh_mano.usd"),
        "root_link": "r_palm",
        "tip_links": ("r_thumb_tip", "r_index_tip", "r_middle_tip", "r_ring_tip", "r_pinky_tip"),
    },
    "mano_left": {
        "urdf_path": os.path.join(ASSETS_DIR, "mano", "lh_mano.urdf"),
        "usd_path": os.path.join(ASSETS_DIR, "mano", "lh_mano", "lh_mano.usd"),
        "root_link": "l_palm",
        "tip_links": ("l_thumb_tip", "l_index_tip", "l_middle_tip", "l_ring_tip", "l_pinky_tip"),
    },
    "allegro_right": {
        "urdf_path": os.path.join(ASSETS_DIR, "allegro_hand", "allegro_hand_right.urdf"),
        "usd_path": os.path.join(ASSETS_DIR, "allegro_hand", "allegro_hand_right", "allegro_hand_right.usd"),
        "root_link": "r_base_link",
        "tip_links": ("r_link_15_tip", "r_link_3_tip", "r_link_7_tip", "r_link_11_tip"),
    },
}


@dataclass(frozen=True)
class JointSpec:
    name: str
    parent: str
    child: str
    kind: str
    origin_transform: torch.Tensor
    axis: Optional[torch.Tensor]
    lower: Optional[float]
    upper: Optional[float]
    mimic_parent: Optional[str]
    mimic_multiplier: float
    mimic_offset: float


def _strip_disallowed_name_attributes(root: et.Element) -> int:
    removed = 0
    for element in root.iter():
        if element.tag in {"visual", "collision"} and "name" in element.attrib:
            del element.attrib["name"]
            removed += 1
    return removed


def load_urdf_silent(urdf_path: str) -> urdf_parser.URDF:
    xml_tree = et.parse(urdf_path)
    root = xml_tree.getroot()
    _strip_disallowed_name_attributes(root)
    sanitized_xml = et.tostring(root, encoding="unicode")
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        return urdf_parser.URDF.from_xml_string(sanitized_xml)


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=torch.float32,
    )


def _make_transform(rotation: torch.Tensor, translation: Sequence[float]) -> torch.Tensor:
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = torch.tensor(list(translation), dtype=torch.float32)
    return transform


def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = F.normalize(axis, dim=-1)
    angle = angle.unsqueeze(-1)
    cos_term = torch.cos(angle)
    sin_term = torch.sin(angle)
    one_minus_cos = 1.0 - cos_term
    x, y, z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
    rotation = torch.zeros(axis.shape[0], 3, 3, dtype=axis.dtype, device=axis.device)
    rotation[:, 0, 0] = (cos_term + x * x * one_minus_cos).squeeze(-1)
    rotation[:, 0, 1] = (x * y * one_minus_cos - z * sin_term).squeeze(-1)
    rotation[:, 0, 2] = (z * x * one_minus_cos + y * sin_term).squeeze(-1)
    rotation[:, 1, 0] = (x * y * one_minus_cos + z * sin_term).squeeze(-1)
    rotation[:, 1, 1] = (cos_term + y * y * one_minus_cos).squeeze(-1)
    rotation[:, 1, 2] = (y * z * one_minus_cos - x * sin_term).squeeze(-1)
    rotation[:, 2, 0] = (z * x * one_minus_cos - y * sin_term).squeeze(-1)
    rotation[:, 2, 1] = (y * z * one_minus_cos + x * sin_term).squeeze(-1)
    rotation[:, 2, 2] = (cos_term + z * z * one_minus_cos).squeeze(-1)
    return rotation


def wrist_dof_to_transform(wrist_dof: torch.Tensor) -> torch.Tensor:
    """Convert (B, 6) wrist DOF [pos(3), axis-angle(3)] to (B, 4, 4) transform."""
    B = wrist_dof.shape[0]
    dtype = wrist_dof.dtype
    device = wrist_dof.device
    pos = wrist_dof[:, :3]
    aa = wrist_dof[:, 3:6]

    angle = torch.linalg.norm(aa, dim=-1)  # (B,)
    axis = aa / (angle.unsqueeze(-1) + 1e-8)
    # For zero rotation, use dummy axis
    mask = angle < 1e-8
    axis[mask] = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
    angle = angle.clone()
    angle[mask] = 0.0

    rot = axis_angle_to_matrix(axis, angle)  # (B, 3, 3)

    T = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = rot
    T[:, :3, 3] = pos
    return T


class HandKinematicsModel:
    """Differentiable FK model for a standalone hand embodiment.

    Mimic joints are excluded from DOF list and auto-computed from their
    parent joint during FK (same as original DexLatent).

    An alignment rotation is applied so all hands face the canonical
    direction (fingers → +x).
    """

    def __init__(
        self,
        hand_name: str,
        urdf_path: str,
        root_link: str,
        tip_links: Sequence[str],
    ) -> None:
        self.hand_name = hand_name
        self.tip_links: List[str] = list(tip_links)
        self.root_link = root_link
        self.urdf = load_urdf_silent(urdf_path)
        self.joint_specs: Dict[str, JointSpec] = {}
        self.children_by_parent: Dict[str, List[str]] = {}
        self.dof_joints: List[str] = []
        self._lower = torch.tensor([], dtype=torch.float32)
        self._upper = torch.tensor([], dtype=torch.float32)
        self._parse_urdf()
        self.traversal_order: List[str] = self._compute_traversal()

        # Static alignment rotation (4x4)
        align_rot = ALIGNMENT_ROTATIONS.get(hand_name, torch.eye(3, dtype=torch.float32))
        self._alignment = torch.eye(4, dtype=torch.float32)
        self._alignment[:3, :3] = align_rot

    def _parse_urdf(self) -> None:
        joints: List[urdf_parser.Joint] = list(self.urdf.joints)
        dof_lowers: List[float] = []
        dof_uppers: List[float] = []
        for joint in joints:
            if joint.origin is not None:
                xyz = joint.origin.xyz if joint.origin.xyz is not None else [0.0, 0.0, 0.0]
                rpy = joint.origin.rpy if joint.origin.rpy is not None else [0.0, 0.0, 0.0]
            else:
                xyz, rpy = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            origin_transform = _make_transform(_rpy_to_matrix(rpy[0], rpy[1], rpy[2]), xyz)

            axis_tensor: Optional[torch.Tensor] = None
            lower: Optional[float] = None
            upper: Optional[float] = None
            mimic_parent: Optional[str] = None
            mimic_multiplier = 1.0
            mimic_offset = 0.0

            if joint.type == "revolute":
                axis_list = joint.axis if joint.axis is not None else [0.0, 0.0, 1.0]
                axis_tensor = torch.tensor(axis_list, dtype=torch.float32)
                axis_tensor = axis_tensor / torch.linalg.norm(axis_tensor)
                if joint.limit is not None:
                    lower = float(joint.limit.lower)
                    upper = float(joint.limit.upper)
                if joint.mimic is not None:
                    mimic_parent = joint.mimic.joint
                    mimic_multiplier = float(joint.mimic.multiplier)
                    mimic_offset = float(joint.mimic.offset)
                else:
                    # Only independent (non-mimic) joints are DOFs
                    self.dof_joints.append(joint.name)
                    dof_lowers.append(lower if lower is not None else 0.0)
                    dof_uppers.append(upper if upper is not None else 0.0)

            self.joint_specs[joint.name] = JointSpec(
                name=joint.name, parent=joint.parent, child=joint.child, kind=joint.type,
                origin_transform=origin_transform, axis=axis_tensor,
                lower=lower, upper=upper, mimic_parent=mimic_parent,
                mimic_multiplier=mimic_multiplier, mimic_offset=mimic_offset,
            )
            self.children_by_parent.setdefault(joint.parent, []).append(joint.name)

        self._lower = torch.tensor(dof_lowers, dtype=torch.float32)
        self._upper = torch.tensor(dof_uppers, dtype=torch.float32)

    def _compute_traversal(self) -> List[str]:
        order: List[str] = []
        queue: List[str] = [self.root_link]
        while queue:
            link = queue.pop(0)
            for joint_name in self.children_by_parent.get(link, []):
                order.append(joint_name)
                queue.append(self.joint_specs[joint_name].child)
        return order

    def dof_count(self) -> int:
        return len(self.dof_joints)

    def joint_name_order(self) -> List[str]:
        return list(self.dof_joints)

    def tip_count(self) -> int:
        return len(self.tip_links)

    def _normalized_to_joint_angles(self, normalized: torch.Tensor) -> Dict[str, torch.Tensor]:
        clipped = torch.clamp(normalized, -1.0, 1.0)
        lower = self._lower.to(device=clipped.device, dtype=clipped.dtype)
        upper = self._upper.to(device=clipped.device, dtype=clipped.dtype)
        angles = (clipped + 1.0) * 0.5 * (upper - lower) + lower
        angle_map: Dict[str, torch.Tensor] = {}
        for index, joint_name in enumerate(self.dof_joints):
            angle_map[joint_name] = angles[:, index]
        # Auto-compute mimic joints from their parent
        for joint_name, spec in self.joint_specs.items():
            if spec.kind == "revolute" and spec.mimic_parent is not None:
                angle_map[joint_name] = (
                    angle_map[spec.mimic_parent] * spec.mimic_multiplier + spec.mimic_offset
                )
        # Clamp all to joint limits
        for joint_name, spec in self.joint_specs.items():
            if spec.kind == "revolute" and joint_name in angle_map:
                lo = float(spec.lower) if spec.lower is not None else float("-inf")
                hi = float(spec.upper) if spec.upper is not None else float("inf")
                angle_map[joint_name] = torch.clamp(angle_map[joint_name], min=lo, max=hi)
        return angle_map

    def expand_to_all_joints(self, normalized_dof: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        """Expand independent DOFs to all revolute joints (including mimic).

        For Isaac Lab visualization: VAE outputs independent DOFs only,
        but Isaac Lab needs all joints.

        Args:
            normalized_dof: (B, D) normalized independent joints.

        Returns:
            all_joint_names: list of all revolute joint names (including mimic)
            all_angles: (B, J) real joint angles in radians for all revolute joints
        """
        angle_map = self._normalized_to_joint_angles(normalized_dof)
        all_names = []
        all_angles_list = []
        for joint_name, spec in self.joint_specs.items():
            if spec.kind == "revolute" and joint_name in angle_map:
                all_names.append(joint_name)
                all_angles_list.append(angle_map[joint_name])
        all_angles = torch.stack(all_angles_list, dim=-1)
        return all_names, all_angles

    def angles_to_normalized(self, angles: torch.Tensor) -> torch.Tensor:
        squeeze = angles.ndim == 1
        batch = angles.unsqueeze(0) if squeeze else angles
        lower = self._lower.to(device=batch.device, dtype=batch.dtype)
        upper = self._upper.to(device=batch.device, dtype=batch.dtype)
        span = upper - lower
        safe_span = torch.where(span.abs() < 1e-8, torch.ones_like(span), span)
        normalized = 2.0 * (batch - lower) / safe_span - 1.0
        normalized = torch.where(span.abs().unsqueeze(0) < 1e-8, torch.zeros_like(normalized), normalized)
        return normalized[0] if squeeze else normalized

    def _forward_internal(
        self,
        normalized_qpos: torch.Tensor,
        wrist_transform: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Run FK with optional wrist transform.

        Args:
            normalized_qpos: (B, D) or (D,) normalized hand joints.
            wrist_transform: (B, 4, 4) optional wrist pose in world frame.
                If None, only alignment rotation is applied.
        """
        squeeze_output = normalized_qpos.ndim == 1
        batch_qpos = normalized_qpos.unsqueeze(0) if squeeze_output else normalized_qpos
        batch = batch_qpos.shape[0]
        dtype = batch_qpos.dtype
        device = batch_qpos.device

        # Root = alignment (static) @ wrist (dynamic, optional)
        alignment = self._alignment.to(dtype=dtype, device=device).unsqueeze(0).expand(batch, -1, -1)
        if wrist_transform is not None:
            root_T = wrist_transform @ alignment
        else:
            root_T = alignment.clone()

        angles_map = self._normalized_to_joint_angles(batch_qpos)
        transforms: Dict[str, torch.Tensor] = {self.root_link: root_T}

        for joint_name in self.traversal_order:
            spec = self.joint_specs[joint_name]
            parent_transform = transforms[spec.parent]
            origin = spec.origin_transform.to(dtype=dtype, device=device).unsqueeze(0).expand(batch, -1, -1)

            if spec.kind == "revolute" and joint_name in angles_map:
                axis = spec.axis.to(dtype=dtype, device=device).unsqueeze(0).expand(batch, -1)
                angle = angles_map[joint_name].to(dtype=dtype, device=device)
                jt = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(batch, 1, 1)
                jt[:, :3, :3] = axis_angle_to_matrix(axis, angle)
            else:
                jt = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).expand(batch, -1, -1)

            transforms[spec.child] = parent_transform @ origin @ jt

        tips = torch.stack([transforms[tip][:, :3, 3] for tip in self.tip_links], dim=1)
        root_out = transforms[self.root_link]
        return tips, root_out, squeeze_output

    def forward(
        self,
        normalized_qpos: torch.Tensor,
        wrist_transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tips, _, squeeze_output = self._forward_internal(normalized_qpos, wrist_transform)
        return tips[0] if squeeze_output else tips

    def forward_with_root_pose(
        self,
        normalized_qpos: torch.Tensor,
        wrist_transform: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tips, root_pose, squeeze_output = self._forward_internal(normalized_qpos, wrist_transform)
        if squeeze_output:
            return tips[0], root_pose[0]
        return tips, root_pose


class MultiHandDifferentiableFK:
    """Registry that builds FK models for a set of hands."""

    def __init__(self, hand_names: Optional[Iterable[str]] = None) -> None:
        targets = list(HAND_CONFIGS.keys()) if hand_names is None else list(hand_names)
        self.models: Dict[str, HandKinematicsModel] = {}
        for name in targets:
            config = HAND_CONFIGS[name]
            self.models[name] = HandKinematicsModel(
                hand_name=name,
                urdf_path=str(config["urdf_path"]),
                root_link=str(config["root_link"]),
                tip_links=tuple(config["tip_links"]),
            )

    def supported_hands(self) -> List[str]:
        return list(self.models.keys())


def solve_inverse_kinematics(
    model: HandKinematicsModel,
    target_positions: torch.Tensor,
    iterations: int = 100,
    learning_rate: float = 0.05,
    initial_qpos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gradient-based IK: optimize normalized joint positions to match fingertip targets."""
    squeeze_batch = target_positions.ndim == 2
    targets = target_positions.unsqueeze(0) if squeeze_batch else target_positions
    dof = model.dof_count()
    batch_size = targets.shape[0]
    dtype = targets.dtype if targets.is_floating_point() else torch.float32
    device = targets.device

    if initial_qpos is None:
        unconstrained_start = torch.zeros(batch_size, dof, dtype=dtype, device=device)
    else:
        seed = initial_qpos.to(dtype=dtype, device=device)
        seed = seed.unsqueeze(0) if seed.ndim == 1 else seed
        seed_clamped = torch.clamp(seed, -1.0 + 1e-6, 1.0 - 1e-6)
        unconstrained_start = torch.atanh(seed_clamped)

    current = unconstrained_start.clone().detach().requires_grad_(True)
    trajectory: List[torch.Tensor] = [torch.tanh(current).detach().clone()]
    optimizer = torch.optim.AdamW([current], lr=learning_rate)
    target = targets.to(dtype=dtype, device=device)

    for _ in range(iterations):
        optimizer.zero_grad()
        normalized = torch.tanh(current)
        tips = model.forward(normalized)
        loss = F.mse_loss(tips, target)
        loss.backward()
        optimizer.step()
        trajectory.append(torch.tanh(current).detach().clone())

    result = torch.stack(trajectory, dim=0)
    return result[:, 0, :] if squeeze_batch else result
