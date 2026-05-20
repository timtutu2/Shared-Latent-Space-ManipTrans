"""
Retargeting 결과를 Isaac Lab viewer에서 재생하여 확인하는 스크립트.

gym_translate/visualize_retargeting.py 의 Isaac Lab 버전.
양손 Shadow Hand + 물체 궤적을 동기화하여 재생합니다.

Usage:
    python maniptrans/anim/view_retarget.py \
        --scene_pkl data/083f7a_demo/scene_01__A001++seq__083f7a577484ba7929a9__2023-04-27-19-25-24.pkl \
        --rh_pkl data/083f7a_demo/shadow/scene_01__A001++seq__083f7a577484ba7929a9__2023-04-27-19-25-24@0_right.pkl \
        --lh_pkl data/083f7a_demo/shadow/scene_01__A001++seq__083f7a577484ba7929a9__2023-04-27-19-25-24@0_left.pkl \
        --obj1_usd data/083f7a_demo/object/S20005/model_align.usd \
        --obj2_usd data/083f7a_demo/object/O02@0094@00004/scan.usd \
        --program_json data/083f7a_demo/gym_translate/scene_01__A001++seq__083f7a577484ba7929a9__2023-04-27-19-25-24.json
"""

# =========================================================================
#  AppLauncher MUST run before any isaaclab import
# =========================================================================
import argparse
import sys

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--device", type=str, default="cuda:0")
_pre_args, _ = _pre_parser.parse_known_args()

from isaaclab.app import AppLauncher
_app_launcher = AppLauncher({"headless": False, "device": _pre_args.device})

import json
import os
import pickle
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation


# =========================================================================
#  Rotation Utilities
# =========================================================================

def aa_to_rotmat(aa):
    """Axis-angle (3,) → rotation matrix (3,3)."""
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3)
    axis = aa / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def aa_to_quat_wxyz(aa):
    """Axis-angle (N,3) numpy → quaternion (N,4) wxyz."""
    angle = np.linalg.norm(aa, axis=-1, keepdims=True).clip(min=1e-8)
    axis = aa / angle
    half = angle / 2.0
    w = np.cos(half)
    xyz = axis * np.sin(half)
    return np.concatenate([w, xyz], axis=-1)


def build_mujoco2gym():
    """Build mujoco→gym 4x4 coordinate transform."""
    m2g = np.eye(4)
    m2g[:3, :3] = (
        aa_to_rotmat(np.array([0, 0, -np.pi / 2]))
        @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
    )
    m2g[:3, 3] = np.array([0, 0, 0.4 + 0.015])  # table surface z
    return m2g


# =========================================================================
#  Hand Config
# =========================================================================

HAND_CONFIGS = {
    "shadow_right": {
        "usd": "data/assets/shadow_hand/shadow_hand_woarm_right/shadow_hand_woarm_right.usd",
        "urdf": "data/assets/shadow_hand/shadow_hand_woarm_right.urdf",
        "stiffness": {"r_(FF|MF|RF|LF|TH)J(3|2|1|0)": 500.0, "r_(LF|TH)J4": 500.0},
        "damping": {"r_(FF|MF|RF|LF|TH)J(3|2|1|0)": 30.0, "r_(LF|TH)J4": 30.0},
    },
    "shadow_left": {
        "usd": "data/assets/shadow_hand/shadow_hand_woarm_left/shadow_hand_woarm_left.usd",
        "urdf": "data/assets/shadow_hand/shadow_hand_woarm_left.urdf",
        "stiffness": {"l_(FF|MF|RF|LF|TH)J(3|2|1|0)": 500.0, "l_(LF|TH)J4": 500.0},
        "damping": {"l_(FF|MF|RF|LF|TH)J(3|2|1|0)": 30.0, "l_(LF|TH)J4": 30.0},
    },
    "inspire_right": {
        "usd": "data/assets/inspire_hand/inspire_hand_right/inspire_hand_right.usd",
        "urdf": "data/assets/inspire_hand/inspire_hand_right.urdf",
        "stiffness": {
            "R_(index|middle|pinky|ring)_(proximal|intermediate)_joint": 500.0,
            "R_thumb_(proximal_yaw|proximal_pitch|intermediate|distal)_joint": 500.0,
        },
        "damping": {
            "R_(index|middle|pinky|ring)_(proximal|intermediate)_joint": 30.0,
            "R_thumb_(proximal_yaw|proximal_pitch|intermediate|distal)_joint": 30.0,
        },
    },
    "artimano_right": {
        "usd": "data/assets/mano/rh_mano_lab/rh_mano_lab.usd",
        "urdf": "data/assets/mano/rh_mano_lab.urdf",
        "stiffness": {
            "r_j_(index|middle|ring|pinky)(2|3)": 500.0,
            "r_j_(thumb|index|middle|ring|pinky)1(y|z)": 500.0,
            "r_j_thumb1x": 500.0,
            "r_j_thumb2(y|z)": 500.0,
            "r_j_thumb3": 500.0,
        },
        "damping": {
            "r_j_(index|middle|ring|pinky)(2|3)": 30.0,
            "r_j_(thumb|index|middle|ring|pinky)1(y|z)": 30.0,
            "r_j_thumb1x": 30.0,
            "r_j_thumb2(y|z)": 30.0,
            "r_j_thumb3": 30.0,
        },
    },
    "artimano_left": {
        "usd": "data/assets/mano/lh_mano/lh_mano.usd",
        "urdf": "data/assets/mano/lh_mano.urdf",
        "stiffness": {
            "l_j_(index|middle|ring|pinky)(2|3)": 500.0,
            "l_j_(thumb|index|middle|ring|pinky)1(y|z)": 500.0,
            "l_j_thumb1x": 500.0,
            "l_j_thumb2(y|z)": 500.0,
            "l_j_thumb3": 500.0,
        },
        "damping": {
            "l_j_(index|middle|ring|pinky)(2|3)": 30.0,
            "l_j_(thumb|index|middle|ring|pinky)1(y|z)": 30.0,
            "l_j_thumb1x": 30.0,
            "l_j_thumb2(y|z)": 30.0,
            "l_j_thumb3": 30.0,
        },
    },
    "allegro_right": {
        "usd": "data/assets/allegro_hand/allegro_hand_right/allegro_hand_right.usd",
        "urdf": "data/assets/allegro_hand/allegro_hand_right.urdf",
        "stiffness": {"r_joint_.*": 300.0},
        "damping": {"r_joint_.*": 20.0},
    },
    "allegro_left": {
        "usd": "data/assets/allegro_hand/allegro_hand_left/allegro_hand_left.usd",
        "urdf": "data/assets/allegro_hand/allegro_hand_left.urdf",
        "stiffness": {"l_joint_.*": 300.0},
        "damping": {"l_joint_.*": 20.0},
    },
}


# =========================================================================
#  DOF Ordering: URDF (Isaac Gym) → PhysX DFS (Isaac Lab)
# =========================================================================

def parse_urdf_joint_order(urdf_path):
    """Parse URDF to get joint declaration order (= Isaac Gym DOF order)."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    names = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type", "") in ("revolute", "prismatic", "continuous"):
            names.append(joint.attrib["name"])
    return names


def parse_urdf_mimic_joints(urdf_path):
    """Parse URDF mimic joint constraints.

    Returns dict: {child_joint_name: (master_joint_name, multiplier, offset)}
    Empty dict if no mimic joints found.
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(urdf_path)
    mimic_info = {}
    for joint in tree.getroot().findall("joint"):
        mimic_elem = joint.find("mimic")
        if mimic_elem is not None:
            child = joint.attrib["name"]
            master = mimic_elem.get("joint")
            mult = float(mimic_elem.get("multiplier", "1.0"))
            off = float(mimic_elem.get("offset", "0.0"))
            mimic_info[child] = (master, mult, off)
    return mimic_info


def parse_urdf_chain_order(urdf_path):
    """Parse URDF to get pytorch_kinematics chain order (= Lab retarget DOF order).

    This is a DFS traversal from root, ordering joints root→leaf per finger.
    Gym retarget uses URDF XML declaration order (leaf→root per finger).
    Lab retarget uses this chain/common order (root→leaf per finger).
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Build parent→children map from joints
    links = [l.attrib['name'] for l in root.findall('link')]
    children = {n: [] for n in links}
    joint_for_child = {}  # child_link → joint_name
    joint_types = {}
    for j in root.findall('joint'):
        jtype = j.attrib.get('type', '')
        parent = j.find('parent').attrib['link']
        child = j.find('child').attrib['link']
        children[parent].append(child)
        joint_for_child[child] = j.attrib['name']
        joint_types[j.attrib['name']] = jtype

    # Find root link (not a child of any joint)
    all_children = set(joint_for_child.keys())
    root_link = None
    for l in links:
        if l not in all_children:
            root_link = l
            break

    # DFS to get joint order (root→leaf)
    dof_order = []
    def dfs(link):
        for child in children[link]:
            jname = joint_for_child[child]
            if joint_types[jname] in ('revolute', 'prismatic', 'continuous'):
                dof_order.append(jname)
            dfs(child)
    dfs(root_link)
    return dof_order


def build_gym2sim_index(gym_names, sim_names):
    """Build gather index: sim_ordered = gym_tensor[:, idx]."""
    idx = []
    for sname in sim_names:
        if sname in gym_names:
            idx.append(gym_names.index(sname))
        else:
            print(f"  [WARN] sim DOF '{sname}' not in gym DOFs")
            idx.append(0)
    return idx


def build_shadow_old_maniptrans_order(side_prefix="r_"):
    """Shadow DOF order used by original ManipTrans Gym retarget pkl."""
    # NOTE: Original ManipTrans ordering is FF -> MF -> RF -> LF -> TH.
    old = [
        "FFJ4", "FFJ3", "FFJ2", "FFJ1",
        "MFJ4", "MFJ3", "MFJ2", "MFJ1",
        "RFJ4", "RFJ3", "RFJ2", "RFJ1",
        "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1",
        "THJ5", "THJ4", "THJ3", "THJ2", "THJ1",
    ]
    converted = []
    for n in old:
        finger = n[:2]
        j = int(n[3:])
        converted.append(f"{side_prefix}{finger}J{j - 1}")
    return converted


def summarize_retarget_pkl(data, label="PKL"):
    """Print concise retarget pkl structure (keys/shapes/dtypes)."""
    print(f"\n  [{label}] structure:")
    if not isinstance(data, dict):
        print(f"    type={type(data)} (not dict)")
        return
    for k, v in data.items():
        shp = getattr(v, "shape", None)
        dt = getattr(v, "dtype", None)
        if shp is not None:
            print(f"    - {k:24s} shape={tuple(shp)} dtype={dt}")
        else:
            print(f"    - {k:24s} type={type(v)}")


def maybe_override_with_lab_fk_dof(pkl_path, data, label="RH"):
    """If sibling *_lab_fk.pkl exists, override opt_dof_pos from that file.

    This fixes Gym-retargeted Shadow files where mesh playback in Isaac Lab can
    diverge even though opt_joints_pos looks correct.
    """
    if pkl_path is None or not pkl_path.endswith(".pkl"):
        return data
    candidates = [pkl_path[:-4] + "_lab_fk.pkl"]
    # Common workspace layout fallback:
    # /.../ManipTrans_Lab/data/...  ->  /.../data/...
    if "/ManipTrans_Lab/data/" in pkl_path:
        p2 = pkl_path.replace("/ManipTrans_Lab/data/", "/data/")
        candidates.append(p2[:-4] + "_lab_fk.pkl")

    alt_path = None
    for c in candidates:
        if os.path.exists(c):
            alt_path = c
            break
    if alt_path is None:
        return data
    try:
        with open(alt_path, "rb") as f:
            alt = pickle.load(f)
        if "opt_dof_pos" not in alt or "opt_dof_pos" not in data:
            return data
        if alt["opt_dof_pos"].shape != data["opt_dof_pos"].shape:
            print(f"  [{label}] lab_fk DOF skipped (shape mismatch): "
                  f"{alt['opt_dof_pos'].shape} vs {data['opt_dof_pos'].shape}")
            return data
        max_diff = float(np.max(np.abs(alt["opt_dof_pos"] - data["opt_dof_pos"])))
        data = dict(data)
        data["opt_dof_pos"] = alt["opt_dof_pos"]
        print(f"  [{label}] Using lab_fk DOF override: {alt_path}")
        print(f"  [{label}] opt_dof_pos max |delta| = {max_diff:.6f} rad")
    except Exception as e:
        print(f"  [{label}] lab_fk DOF override failed: {e}")
    return data


# =========================================================================
#  Viewer
# =========================================================================

class RetargetViewer:
    """Isaac Lab viewer for bimanual retargeted Shadow Hands + objects."""

    def __init__(self, device, rh_type="shadow_right", lh_type="shadow_left",
                 obj_usd_paths=None, lab_retarget=False, shadow_pkl_order="auto"):
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.sim.schemas import FixedTendonPropertiesCfg, JointDrivePropertiesCfg
        from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
        from pxr import UsdPhysics, Usd

        self.device = device
        self._hands = {}       # "rh" / "lh" → Articulation
        self._dof_idx = {}     # "rh" / "lh" → gather index list
        self._obj_xform_ops = {}  # prim_path → xform op (cached)
        self._lab_retarget = lab_retarget
        self._shadow_pkl_order = shadow_pkl_order

        # ---- Simulator ----
        sim_cfg = sim_utils.SimulationCfg(device=device, dt=1.0/120, render_interval=4)
        sim_cfg.physx.bounce_threshold_velocity = 0.2
        sim_cfg.physx.max_position_iteration_count = 4
        sim_cfg.physx.max_velocity_iteration_count = 0
        sim_cfg.physics_material.static_friction = 4.0
        sim_cfg.physics_material.dynamic_friction = 4.0
        self._sim = sim_utils.SimulationContext(sim_cfg)
        self._stage = self._sim._stage  # for USD operations

        # ---- Ground ----
        ground_cfg = GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0))
        spawn_ground_plane("/World/ground", ground_cfg)

        # ---- Env ----
        self._stage.DefinePrim("/World/envs/env_0", "Xform")

        # ---- Hands ----
        hand_types = [(l, t) for l, t in [("rh", rh_type), ("lh", lh_type)] if t is not None]
        self._hand_types = hand_types
        for label, cfg_key in hand_types:
            cfg = HAND_CONFIGS[cfg_key]
            art_props = sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.0, stabilization_threshold=0.001,
                fix_root_link=False)
            rigid_props = sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=5.0,
                linear_damping=20.0, angular_damping=20.0,
                max_linear_velocity=50.0, max_angular_velocity=100.0)
            fixed_tendons = FixedTendonPropertiesCfg(
                tendon_enabled=False, stiffness=None, damping=0.1,
                limit_stiffness=30.0, offset=None, rest_length=None)
            joint_drive = JointDrivePropertiesCfg(
                drive_type="force", max_effort=None,
                max_velocity=None, stiffness=None, damping=None)
            collision_props = sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0)

            usd_cfg = sim_utils.UsdFileCfg(
                usd_path=cfg["usd"],
                articulation_props=art_props,
                fixed_tendons_props=fixed_tendons,
                joint_drive_props=joint_drive,
                rigid_props=rigid_props,
                collision_props=collision_props,
                activate_contact_sensors=False)

            prim_path = f"/World/envs/env_0/{label}"
            usd_cfg.func(prim_path, usd_cfg,
                         translation=(0.0, 0.0, 0.0),
                         orientation=(1.0, 0.0, 0.0, 0.0))

            actuator = ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=cfg["stiffness"],
                damping=cfg["damping"])

            art_cfg = ArticulationCfg(
                prim_path=f"/World/envs/env_.*/{label}", spawn=None,
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
                actuators={"act": actuator})
            self._hands[label] = Articulation(art_cfg)

        # ---- Objects (visual-only, physics stripped) ----
        if obj_usd_paths:
            for i, usd_path in enumerate(obj_usd_paths):
                if usd_path and os.path.exists(usd_path):
                    obj_prim = f"/World/envs/env_0/obj{i}"
                    obj_cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
                    obj_cfg.func(obj_prim, obj_cfg,
                                 translation=(0.0, 0.0, 0.0),
                                 orientation=(1.0, 0.0, 0.0, 0.0))
                    self._strip_physics(obj_prim)
                    print(f"  [obj{i}] Loaded: {usd_path}")

        # ---- Filter collisions ----
        from isaacsim.core.cloner import Cloner
        from pxr import PhysxSchema
        cloner = Cloner(self._stage)
        physics_scene = None
        for prim in self._stage.Traverse():
            if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                physics_scene = prim.GetPrimPath().pathString
                break
        if physics_scene:
            cloner.filter_collisions(physics_scene, "/World/collisions",
                                     ["/World/envs/env_0"], global_paths=[])

        # ---- Init ----
        self._sim.reset()

        # ---- Build DOF mappings + detect palm body index ----
        self._palm_idx = {}  # label → palm body index in PhysX order
        self._mimic_pairs = {}  # label → [(child_sim_idx, master_sim_idx, mult, off)]
        self._body_order_sim2common = {}  # label → list[int], PhysX body idx → common body idx
        self._inspire_remap = {}  # label → {dst_sim_idx: src_sim_idx} or None
        self._pkl_dof_names_map = {}  # label -> pkl dof names used for mapping
        for label, cfg_key in hand_types:
            cfg = HAND_CONFIGS[cfg_key]
            meta = self._hands[label].root_physx_view.shared_metatype
            link_names = meta.link_names
            sim_names = list(meta.dof_names) if hasattr(meta, 'dof_names') else []
            if cfg_key.startswith("shadow") and not lab_retarget:
                # In practice for most generated *_sv_dict.pkl files in this repo,
                # DOF vectors follow URDF XML declaration order.
                # Keep legacy ManipTrans order as an explicit opt-in switch.
                if shadow_pkl_order == "maniptrans_old":
                    side_prefix = "r_" if cfg_key.endswith("right") else "l_"
                    pkl_dof_names = build_shadow_old_maniptrans_order(side_prefix=side_prefix)
                    print(f"  [{label}] Shadow pkl order: maniptrans_old")
                elif os.path.exists(cfg["urdf"]):
                    pkl_dof_names = parse_urdf_joint_order(cfg["urdf"])
                    mode = "auto->urdf_xml" if shadow_pkl_order == "auto" else "urdf_xml"
                    print(f"  [{label}] Shadow pkl order: {mode}")
                else:
                    pkl_dof_names = sim_names
            elif os.path.exists(cfg["urdf"]):
                if lab_retarget:
                    # Lab retarget: DOF in chain/common order (root→leaf DFS)
                    pkl_dof_names = parse_urdf_chain_order(cfg["urdf"])
                else:
                    # Gym retarget: DOF in URDF XML declaration order (leaf→root)
                    pkl_dof_names = parse_urdf_joint_order(cfg["urdf"])
            else:
                pkl_dof_names = sim_names
            self._dof_idx[label] = build_gym2sim_index(pkl_dof_names, sim_names)
            self._pkl_dof_names_map[label] = list(pkl_dof_names)
            same = (pkl_dof_names == sim_names)

            # Find palm body index
            palm_idx = 0
            for i, n in enumerate(link_names):
                if "palm" in n.lower() or "base_link" in n.lower():
                    palm_idx = i
                    break
            self._palm_idx[label] = palm_idx

            # ---- Build body order: sim (PhysX DFS) → common (palm-rooted DFS) ----
            # Same logic as mano2dexhand_lab.py _build_order_tensors
            link_parent_indices = meta.link_parent_indices
            num_links = len(link_names)
            link_children = [[] for _ in range(num_links)]
            for ln in link_names:
                if ln in link_parent_indices:
                    pid = link_parent_indices[ln]
                    lid = list(link_names).index(ln)
                    link_children[pid].append(lid)

            # Find PhysX root
            root_id = 0
            for ii, ln in enumerate(link_names):
                if ln not in link_parent_indices:
                    root_id = ii
                    break

            # Re-root at palm if needed
            if palm_idx != root_id:
                if palm_idx in link_children[root_id]:
                    link_children[root_id].remove(palm_idx)
                link_children[palm_idx].append(root_id)
                root_id = palm_idx

            body_order = []
            def _dfs(lid):
                body_order.append(lid)
                for ch in link_children[lid]:
                    _dfs(ch)
            _dfs(root_id)
            self._body_order_sim2common[label] = body_order
            # common_body_names: body names in the order matching pkl's opt_joints_pos
            common_body_names = [link_names[i] for i in body_order]
            print(f"  [{label}] DOFs: pkl={len(pkl_dof_names)} sim={len(sim_names)} same={same}, "
                  f"palm_idx={palm_idx} ({'ROOT==PALM' if palm_idx == 0 else 'ROOT!=PALM'})")
            print(f"  [{label}] body_order_sim2common (full):")
            for bi, si in enumerate(body_order):
                print(f"    common[{bi:2d}] = sim[{si:2d}] = {common_body_names[bi]}")
            print(f"  [{label}] pkl_dof_names: {pkl_dof_names}")
            print(f"  [{label}] sim_dof_names: {sim_names}")
            print(f"  [{label}] dof_idx (pkl→sim gather): {self._dof_idx[label]}")

            # ---- Build mimic constraint map (sim DOF index → sim DOF index) ----
            mimic_info = parse_urdf_mimic_joints(cfg["urdf"])
            mimic_pairs = []  # [(child_sim_idx, master_sim_idx, multiplier, offset)]
            for child_name, (master_name, mult, off) in mimic_info.items():
                child_idx = master_idx = None
                for si, sn in enumerate(sim_names):
                    if sn == child_name or sn.endswith(child_name):
                        child_idx = si
                    if sn == master_name or sn.endswith(master_name):
                        master_idx = si
                if child_idx is not None and master_idx is not None:
                    mimic_pairs.append((child_idx, master_idx, mult, off))
            self._mimic_pairs[label] = mimic_pairs
            if mimic_pairs:
                print(f"  [{label}] Mimic constraints ({len(mimic_pairs)}):")
                for ci, mi, mult, off in mimic_pairs:
                    print(f"    sim[{ci}]({sim_names[ci]}) = "
                          f"{mult}*sim[{mi}]({sim_names[mi]}) + {off}")

            # ---- Inspire DOF remap (USD joint names don't match physical fingers) ----
            # sim_joint_name = actual_pkl_joint_value_to_use
            _INSPIRE_REMAP = {
                "R_thumb_proximal_yaw_joint":   "R_ring_proximal_joint",
                "R_thumb_proximal_pitch_joint":  "R_thumb_distal_joint",
                "R_index_proximal_joint":        "R_thumb_proximal_yaw_joint",
                "R_middle_proximal_joint":       "R_thumb_intermediate_joint",
                "R_pinky_proximal_joint":        "R_index_proximal_joint",
                "R_ring_proximal_joint":         "R_middle_proximal_joint",
                "R_thumb_intermediate_joint":    "R_ring_intermediate_joint",
                "R_index_intermediate_joint":    "R_thumb_proximal_pitch_joint",
                "R_middle_intermediate_joint":   "R_pinky_intermediate_joint",
                "R_pinky_intermediate_joint":    "R_index_intermediate_joint",
                "R_ring_intermediate_joint":     "R_middle_intermediate_joint",
                "R_thumb_distal_joint":          "R_pinky_proximal_joint",
            }
            if "inspire" in cfg_key:
                remap = {}
                for dst_name, src_name in _INSPIRE_REMAP.items():
                    dst_idx = src_idx = None
                    for si, sn in enumerate(sim_names):
                        if sn.endswith(dst_name) or sn == dst_name:
                            dst_idx = si
                        if sn.endswith(src_name) or sn == src_name:
                            src_idx = si
                    if dst_idx is not None and src_idx is not None and dst_idx != src_idx:
                        remap[dst_idx] = src_idx
                self._inspire_remap[label] = remap
                print(f"  [{label}] Inspire DOF remap ({len(remap)} entries):")
                for dst, src in sorted(remap.items()):
                    print(f"    sim[{dst}]({sim_names[dst]}) ← sim[{src}]({sim_names[src]})")
            else:
                self._inspire_remap[label] = None

        # ---- Camera ----
        self._sim.set_camera_view(
            eye=np.array([0.5, -0.3, 0.7]),
            target=np.array([0.0, 0.0, 0.45]))

        # ---- Debug draw (for MANO skeleton overlay) ----
        from isaacsim.util.debug_draw import _debug_draw
        self._draw = _debug_draw.acquire_debug_draw_interface()

        print("[RetargetViewer] Ready")

    def _correct_root_for_palm(self, label, desired_pos, desired_rot_wxyz, dof_sim):
        """When PhysX root != palm, compute the root pose that places palm at desired pose.

        Same logic as mano2dexhand_lab.py _correct_root_pose_for_palm.
        """
        palm_idx = self._palm_idx[label]
        if palm_idx == 0:
            # Root IS palm, no correction needed
            root_pose = torch.zeros(1, 7, device=self.device)
            root_pose[0, :3] = desired_pos
            root_pose[0, 3:] = desired_rot_wxyz
            return root_pose

        hand = self._hands[label]
        env_ids = torch.tensor([0], device=self.device, dtype=torch.int32)

        # Step 1: place root at identity, set DOFs, step, read palm offset
        identity_pose = torch.zeros(1, 7, device=self.device)
        identity_pose[0, 3] = 1.0  # identity quat wxyz
        zero_vel = torch.zeros(1, 6, device=self.device)

        hand.write_root_link_pose_to_sim(identity_pose, env_ids)
        hand.write_root_link_velocity_to_sim(zero_vel, env_ids)
        hand.write_joint_position_to_sim(dof_sim, env_ids=env_ids)
        hand.write_joint_velocity_to_sim(torch.zeros_like(dof_sim), env_ids=env_ids)
        hand.reset(env_ids)
        hand.write_data_to_sim()
        self._sim.step(render=False)
        hand.update(self._sim.get_physics_dt())

        # Read actual positions
        body_poses = hand.data.body_link_pose_w  # (1, B, 7)
        actual_root_pos = body_poses[0, 0, :3]
        actual_root_rot = body_poses[0, 0, 3:]  # wxyz
        actual_palm_pos = body_poses[0, palm_idx, :3]
        actual_palm_rot = body_poses[0, palm_idx, 3:]  # wxyz

        # Step 2: compute corrected root pose
        palm_conj = actual_palm_rot.clone()
        palm_conj[1:] = -palm_conj[1:]

        rot_correction = self._quat_mul(desired_rot_wxyz, palm_conj)
        corrected_root_rot = self._quat_mul(rot_correction, actual_root_rot)

        delta_pos = actual_root_pos - actual_palm_pos
        rotated_delta = self._quat_rotate(rot_correction, delta_pos)
        corrected_root_pos = desired_pos + rotated_delta

        root_pose = torch.zeros(1, 7, device=self.device)
        root_pose[0, :3] = corrected_root_pos
        root_pose[0, 3:] = corrected_root_rot
        return root_pose

    @staticmethod
    def _quat_mul(q1, q2):
        """Quaternion multiply (wxyz)."""
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    @staticmethod
    def _quat_rotate(q, v):
        """Rotate vector v by quaternion q (wxyz)."""
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        t = 2.0 * torch.stack([
            qy*v[2] - qz*v[1],
            qz*v[0] - qx*v[2],
            qx*v[1] - qy*v[0],
        ])
        return v + qw*t + torch.stack([
            qy*t[2] - qz*t[1],
            qz*t[0] - qx*t[2],
            qx*t[1] - qy*t[0],
        ])

    def _strip_physics(self, prim_path):
        """Remove all physics APIs from prim subtree (visual-only)."""
        from pxr import UsdPhysics, PhysxSchema, Usd
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        for p in Usd.PrimRange(prim):
            for api in [UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI,
                        UsdPhysics.ArticulationRootAPI]:
                if p.HasAPI(api):
                    p.RemoveAPI(api)
            try:
                if p.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    p.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
            except Exception:
                pass

    def _set_obj_xform(self, prim_path, mat4x4):
        """Set visual-only prim transform using 4x4 matrix."""
        from pxr import Gf, UsdGeom

        if prim_path not in self._obj_xform_ops:
            prim = self._stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()
            self._obj_xform_ops[prim_path] = xformable.AddTransformOp()

        # USD uses row-vector convention (translation in last row),
        # numpy uses column-vector convention (translation in last column).
        # Transpose to convert.
        m = mat4x4.astype(float).T
        gf_mat = Gf.Matrix4d(
            m[0,0], m[0,1], m[0,2], m[0,3],
            m[1,0], m[1,1], m[1,2], m[1,3],
            m[2,0], m[2,1], m[2,2], m[2,3],
            m[3,0], m[3,1], m[3,2], m[3,3])
        self._obj_xform_ops[prim_path].Set(gf_mat)

    # ---- MANO skeleton ----

    # MANO 778-vertex model: key vertex indices for skeleton visualization
    MANO_SKELETON_VIDS = [
        0,    # 0: wrist
        737,  # 1: thumb_base
        739,  # 2: thumb_mid
        744,  # 3: thumb_tip
        36,   # 4: index_base
        301,  # 5: index_mid
        320,  # 6: index_tip
        149,  # 7: middle_base
        425,  # 8: middle_mid
        443,  # 9: middle_tip
        467,  # 10: ring_base
        538,  # 11: ring_mid
        555,  # 12: ring_tip
        600,  # 13: pinky_base
        657,  # 14: pinky_mid
        672,  # 15: pinky_tip
    ]
    MANO_BONES = [
        (0,1),(1,2),(2,3),      # thumb
        (0,4),(4,5),(5,6),      # index
        (0,7),(7,8),(8,9),      # middle
        (0,10),(10,11),(11,12), # ring
        (0,13),(13,14),(14,15), # pinky
    ]

    def load_mano_skeleton(self, sv_dict_path):
        """Load MANO vertices → skeleton keypoints with mujoco2gym transform."""
        sv = np.load(sv_dict_path, allow_pickle=True).item()
        verts = sv["rhand_verts"]  # (T, 778, 3)
        kp = verts[:, self.MANO_SKELETON_VIDS, :]  # (T, 16, 3)

        # Apply mujoco2gym transform
        m2g = build_mujoco2gym()
        R, t = m2g[:3, :3], m2g[:3, 3]
        kp_flat = kp.reshape(-1, 3)
        kp_transformed = (R @ kp_flat.T).T + t
        return kp_transformed.reshape(kp.shape).astype(np.float32)

    def _draw_skeleton(self, keypoints, bones=None):
        """Draw skeleton for one frame using debug_draw."""
        self._draw.clear_points()
        self._draw.clear_lines()
        if bones is None:
            bones = self.MANO_BONES

        # Bones (green lines)
        starts, ends, lcolors, lsizes = [], [], [], []
        for (i, j) in bones:
            starts.append(keypoints[i].tolist())
            ends.append(keypoints[j].tolist())
            lcolors.append((0.0, 1.0, 0.0, 1.0))
            lsizes.append(3.0)
        self._draw.draw_lines(starts, ends, lcolors, lsizes)

        # Joint points (blue), wrist point (red)
        pts = [keypoints[i].tolist() for i in range(len(keypoints))]
        colors = [(0.0, 0.4, 1.0, 1.0)] * len(pts)
        sizes = [10.0] * len(pts)
        if len(pts) > 0:
            colors[0] = (1.0, 0.1, 0.1, 1.0)  # wrist
            sizes[0] = 14.0
        self._draw.draw_points(pts, colors, sizes)

    # Bone links per hand type (same as *_env.py bone_links)
    BONE_LINKS = {
        "shadow": [
            (0,1),(0,6),(0,12),(0,17),(0,22),
            (2,3),(3,4),(4,5),
            (7,8),(8,9),(9,10),(10,11),
            (13,14),(14,15),(15,16),
            (18,19),(19,20),(20,21),
            (23,24),(24,25),(25,26),(26,27),
        ],
        "inspire": [
            (0,1),(0,4),(0,7),(0,10),(0,13),
            (13,14),(3,2),(2,1),
            (6,5),(5,4),
            (9,8),(8,7),
            (12,11),(11,10),
            (17,16),(16,15),(15,14),
        ],
        "artimano": [],  # TODO: add if needed
        "allegro": [
            (0, 1), (0, 6), (0, 11), (0, 16),
            (2, 3), (3, 4), (4, 5),
            (7, 8), (8, 9), (9, 10),
            (12, 13), (13, 14), (14, 15),
            (17, 18), (18, 19), (19, 20),
        ],
    }

    # Preferred named links (converted to index pairs via meta_body_names when available).
    BONE_LINKS_BY_NAME = {
        "allegro": [
            ("base_link", "link_0"), ("base_link", "link_4"), ("base_link", "link_8"), ("base_link", "link_12"),
            ("link_0", "link_1"), ("link_1", "link_2"), ("link_2", "link_3"), ("link_3", "link_3_tip"),
            ("link_4", "link_5"), ("link_5", "link_6"), ("link_6", "link_7"), ("link_7", "link_7_tip"),
            ("link_8", "link_9"), ("link_9", "link_10"), ("link_10", "link_11"), ("link_11", "link_11_tip"),
            ("link_12", "link_13"), ("link_13", "link_14"), ("link_14", "link_15"), ("link_15", "link_15_tip"),
        ],
    }

    @staticmethod
    def _canonical_body_name(name: str) -> str:
        s = name.strip()
        for p in ("r_", "l_", "R_", "L_"):
            if s.startswith(p):
                s = s[len(p):]
                break
        return s.replace(".", "_").lower()

    def load_retarget_skeleton(self, pkl_data, dexhand="shadow"):
        """Extract skeleton keypoints from retarget pkl's opt_joints_pos.

        Returns (T, B, 3) and bone list for the given hand type.
        """
        joints = pkl_data["opt_joints_pos"]
        bones = self.BONE_LINKS.get(dexhand, [])

        # Rebuild links from meta_body_names when available (more robust than fixed index order).
        if dexhand in self.BONE_LINKS_BY_NAME and "meta_body_names" in pkl_data:
            body_names = list(pkl_data.get("meta_body_names", []))
            if len(body_names) == joints.shape[1]:
                canon_to_idx = {}
                for i, bn in enumerate(body_names):
                    canon_to_idx[self._canonical_body_name(bn)] = i
                rebuilt = []
                for a, b in self.BONE_LINKS_BY_NAME[dexhand]:
                    ca = self._canonical_body_name(a)
                    cb = self._canonical_body_name(b)
                    if ca in canon_to_idx and cb in canon_to_idx:
                        rebuilt.append((canon_to_idx[ca], canon_to_idx[cb]))
                if rebuilt:
                    bones = rebuilt
        return joints, bones

    def play(self, rh_data, lh_data, obj_trajs, fps=30, loop=True, speed=1,
             retarget_skeleton=None, retarget_bones=None, physics_playback=False):
        """
        Playback retargeted hands + object trajectories.

        Args:
            rh_data: dict with opt_wrist_pos (T,3), opt_wrist_rot (T,3), opt_dof_pos (T,D)
            lh_data: same for left hand (or None)
            obj_trajs: list of (T, 4, 4) numpy arrays for each object (already transformed)
            fps: playback frame rate
            loop: loop playback
            speed: playback speed multiplier
            retarget_skeleton: (T, 28, 3) retarget body positions (optional)
            retarget_bones: list of (i,j) bone pairs for skeleton drawing
        """
        # ---- Prepare hand tensors ----
        hands_data = {}
        T = 0
        for label, data in [("rh", rh_data), ("lh", lh_data)]:
            if data is None:
                continue
            # Use opt_joints_pos[:,0,:] (palm) as wrist position if available
            if "opt_joints_pos" in data:
                wrist_pos = data["opt_joints_pos"][:, 0, :]
            else:
                wrist_pos = data["opt_wrist_pos"]
            wrist_rot = data["opt_wrist_rot"]
            dof_pos = data["opt_dof_pos"]
            T_hand = wrist_pos.shape[0]
            if T == 0:
                T = T_hand
            else:
                T = min(T, T_hand)

            wrist_quat = aa_to_quat_wxyz(wrist_rot)  # (T, 4) wxyz
            idx = self._dof_idx[label]

            hands_data[label] = {
                "wrist_pos": torch.tensor(wrist_pos, device=self.device, dtype=torch.float32),
                "wrist_quat": torch.tensor(wrist_quat, device=self.device, dtype=torch.float32),
                "dof_pos": torch.tensor(dof_pos, device=self.device, dtype=torch.float32),
                "idx": idx,
            }

        # Clamp object trajectories to T
        for i, traj in enumerate(obj_trajs):
            if traj is not None and traj.shape[0] > T:
                obj_trajs[i] = traj[:T]

        env_ids = torch.tensor([0], device=self.device, dtype=torch.int32)
        frame_dt = 1.0 / fps

        print(f"\n  Playing {T} frames at {fps}fps (speed={speed}x, loop={'on' if loop else 'off'})")
        print(f"  Hands: {list(hands_data.keys())}, Objects: {sum(1 for t in obj_trajs if t is not None)}")
        print(f"  Press Ctrl+C to stop\n")

        try:
            loop_count = 0
            while True:
                for t in range(0, T, speed):
                    t0 = time.time()

                    # ---- Set hand poses (before sim.step) ----
                    for label, hd in hands_data.items():
                        dof_sim = hd["dof_pos"][t:t+1, hd["idx"]]

                        # Inspire DOF remap: USD joint names don't match physical fingers
                        if self._inspire_remap.get(label):
                            orig = dof_sim.clone()
                            for dst, src in self._inspire_remap[label].items():
                                dof_sim[:, dst] = orig[:, src]

                        root_vel = torch.zeros(1, 6, device=self.device)

                        # Correct root pose when PhysX root != palm
                        root_pose = self._correct_root_for_palm(
                            label, hd["wrist_pos"][t], hd["wrist_quat"][t], dof_sim)

                        hand = self._hands[label]
                        hand.write_root_link_pose_to_sim(root_pose, env_ids)
                        hand.write_root_link_velocity_to_sim(root_vel, env_ids)
                        hand.set_joint_position_target(dof_sim)
                        hand.write_joint_position_to_sim(dof_sim, env_ids=env_ids)
                        hand.write_joint_velocity_to_sim(
                            torch.zeros_like(dof_sim), env_ids=env_ids)

                    if physics_playback:
                        # ---- Capture DOF BEFORE physics step (frame 0 only) ----
                        _dof_before = {}
                        if t == 0 and loop_count == 0:
                            for label in hands_data:
                                hand = self._hands[label]
                                hand.write_data_to_sim()
                                # Force a read-back without stepping
                                _dof_before[label] = hand.data.joint_pos[0, :].clone()

                        # ---- Step physics ----
                        for label in hands_data:
                            self._hands[label].write_data_to_sim()
                        self._sim.step(render=False)
                        sim_dt = self._sim.get_physics_dt()
                        for label in hands_data:
                            self._hands[label].update(sim_dt)

                        # ---- Compare DOF BEFORE vs AFTER physics step ----
                        if t == 0 and loop_count == 0 and _dof_before:
                            for label in hands_data:
                                hand = self._hands[label]
                                meta = hand.root_physx_view.shared_metatype
                                sim_dof_names = list(meta.dof_names) if hasattr(meta, 'dof_names') else []
                                dof_after = hand.data.joint_pos[0, :]
                                dof_bef = _dof_before[label]
                                print(f"\n  [{label}] === BEFORE vs AFTER sim.step() ===")
                                print(f"  {'sim_idx':>7s} {'sim_dof_name':>40s}  {'before':>9s}  {'after':>9s}  {'diff':>9s}")
                                for si in range(len(sim_dof_names)):
                                    b = dof_bef[si].item()
                                    a = dof_after[si].item()
                                    d = abs(b - a)
                                    flag = " <<<" if d > 0.001 else ""
                                    print(f"  {si:7d} {sim_dof_names[si]:>40s}  {b:9.5f}  {a:9.5f}  {d:9.5f}{flag}")

                    # ---- Set object xforms AFTER physics step ----
                    for i, traj in enumerate(obj_trajs):
                        if traj is not None and t < traj.shape[0]:
                            self._set_obj_xform(f"/World/envs/env_0/obj{i}", traj[t])

                    # ---- Draw retarget skeleton overlay ----
                    if retarget_skeleton is not None and t < retarget_skeleton.shape[0]:
                        self._draw_skeleton(retarget_skeleton[t], retarget_bones)

                    # ---- Debug: print applied values at frame 0 ----
                    if t == 0 and loop_count == 0:
                        for label, hd in hands_data.items():
                            hand = self._hands[label]
                            meta = hand.root_physx_view.shared_metatype
                            sim_dof_names = list(meta.dof_names) if hasattr(meta, 'dof_names') else []
                            # Use the exact pkl DOF list that produced `hd["idx"]`.
                            pkl_dof_names = self._pkl_dof_names_map.get(label, sim_dof_names)

                            print(f"\n  [{label}] ========== Frame 0 DOF Debug ==========")
                            print(f"\n  [CHECK 1] DOF 이름 매핑: sim 각 위치에 pkl 어떤 값이 들어가는지")
                            print(f"  {'sim_idx':>7s} {'sim_dof_name':>40s}  ←  {'pkl_dof_name':>40s}  {'match':>8s}")
                            idx = hd["idx"]
                            for si in range(len(sim_dof_names)):
                                pi = idx[si]  # pkl index for this sim position
                                sim_n = sim_dof_names[si]
                                pkl_n = pkl_dof_names[pi] if pi < len(pkl_dof_names) else "?"
                                match = "OK" if (pkl_n == sim_n or pkl_n.endswith(sim_n) or sim_n.endswith(pkl_n)) else "MISMATCH"
                                print(f"  {si:7d} {sim_n:>40s}  ←  pkl[{pi:2d}] {pkl_n:>40s}  {match:>8s}")

                            print(f"\n  [CHECK 2] DOF 값 비교: pkl원본 vs reindex후(sim에 전달) vs PhysX실제적용")
                            pkl_dof_raw = hd["dof_pos"][0].cpu().numpy()
                            dof_reindexed = hd["dof_pos"][0, idx].cpu().numpy()
                            actual_dof = hand.data.joint_pos[0, :].cpu().numpy()
                            print(f"  {'sim_idx':>7s} {'sim_dof_name':>40s}  {'pkl→sim':>9s}  {'physx':>9s}  {'diff':>9s}  {'note':>10s}")
                            mimic_child_indices = {ci for ci, mi, mult, off in self._mimic_pairs.get(label, [])}
                            for si in range(len(sim_dof_names)):
                                sent = dof_reindexed[si] if si < len(dof_reindexed) else 0
                                got = actual_dof[si] if si < len(actual_dof) else 0
                                diff = abs(sent - got)
                                note = ""
                                if si in mimic_child_indices:
                                    note = "MIMIC"
                                if diff > 0.001:
                                    note += " <<<DIFF"
                                sn = sim_dof_names[si]
                                print(f"  {si:7d} {sn:>40s}  {sent:9.5f}  {got:9.5f}  {diff:9.5f}  {note:>10s}")

                            print(f"\n  [CHECK 3] Wrist pose")
                            print(f"  desired wrist_pos:       {hd['wrist_pos'][0].cpu().numpy()}")
                            print(f"  desired wrist_quat(wxyz): {hd['wrist_quat'][0].cpu().numpy()}")
                            actual_root = hand.data.body_link_pose_w[0, 0, :]
                            print(f"  actual root pos:         {actual_root[:3].cpu().numpy()}")
                            print(f"  actual root quat:        {actual_root[3:].cpu().numpy()}")
                            print(f"  ==========================================")

                    # ---- Debug: compare skeleton vs articulation body positions ----
                    if t == 0 and loop_count == 0 and retarget_skeleton is not None:
                        for label in hands_data:
                            hand = self._hands[label]
                            body_pos_sim = hand.data.body_link_pose_w[0, :, :3]
                            meta = hand.root_physx_view.shared_metatype
                            link_names = meta.link_names
                            skel = retarget_skeleton[t]
                            num_skel_bodies = skel.shape[0]

                            # Reorder articulation bodies to common order (same as pkl)
                            body_order = self._body_order_sim2common.get(label, list(range(len(link_names))))
                            body_pos_common = body_pos_sim[body_order]
                            common_names = [link_names[i] for i in body_order]

                            print(f"\n  [{label}] Skeleton vs Articulation (frame 0, {num_skel_bodies} bodies, common order):")
                            print(f"  {'idx':>4s} {'body_name':>24s} {'skel_pos':>30s}  {'artic_pos':>30s}  {'dist':>8s}")
                            for gi in range(min(num_skel_bodies, len(common_names))):
                                skel_p = skel[gi]
                                art_p = body_pos_common[gi].cpu().numpy()
                                dist = np.linalg.norm(skel_p - art_p)
                                flag = " <<<" if dist > 0.01 else ""
                                bname = common_names[gi] if gi < len(common_names) else f"body_{gi}"
                                print(f"  {gi:4d} {bname:20s} [{skel_p[0]:8.4f},{skel_p[1]:8.4f},{skel_p[2]:8.4f}]"
                                      f"  [{art_p[0]:8.4f},{art_p[1]:8.4f},{art_p[2]:8.4f}]  {dist:8.5f}{flag}")

                    # ---- Render ----
                    self._sim.render()

                    elapsed = time.time() - t0
                    if elapsed < frame_dt:
                        time.sleep(frame_dt - elapsed)

                    if t % 30 == 0:
                        print(f"  frame {t:4d}/{T}", end="\r")

                print(f"  frame {T:4d}/{T} -- done")
                if not loop:
                    break
                loop_count += 1
                print("  Looping...")

        except KeyboardInterrupt:
            print("\n  Stopped by user")

        self._sim.stop()


# =========================================================================
#  Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="View retargeting (Isaac Lab)")

    # Data
    parser.add_argument("--scene_pkl", type=str, default=None,
                        help="Original scene pkl (for object trajectories, bimanual mode)")
    parser.add_argument("--rh_pkl", type=str, default=None,
                        help="Right hand retarget pkl")
    parser.add_argument("--lh_pkl", type=str, default=None,
                        help="Left hand retarget pkl")

    # Objects
    parser.add_argument("--obj1_usd", type=str, default=None)
    parser.add_argument("--obj2_usd", type=str, default=None)
    parser.add_argument("--obj3_usd", type=str, default=None)
    parser.add_argument("--obj_traj", type=str, default=None,
                        help="Object trajectory pkl (obj_pos + obj_rot, for single-object mode)")
    parser.add_argument("--obj2_traj", type=str, default=None,
                        help="Second object trajectory pkl (obj_pos + obj_rot)")

    # Hand type
    parser.add_argument("--dexhand", type=str, default="shadow",
                        choices=["shadow", "inspire", "artimano", "allegro"],
                        help="Dexterous hand type (default: shadow)")

    # Program info (frame range)
    parser.add_argument("--program_json", type=str, default=None,
                        help="Program info JSON (for frame range)")
    parser.add_argument("--stage", type=int, default=0,
                        help="Program stage index")

    # Retarget source
    parser.add_argument("--lab_retarget", action="store_true",
                        help="PKL is from mano2dexhand_lab.py (DOF in chain/common order). "
                             "Default: Gym retarget (DOF in URDF XML order)")

    # Playback
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--no_loop", action="store_true")
    parser.add_argument("--subsample", type=int, default=4,
                        help="MoCap subsample factor (120Hz→30Hz = 4)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--physics_playback", action="store_true",
                        help="If set, run one physics step per frame. "
                             "Default is kinematic playback (no physics step) for exact overlay.")
    parser.add_argument("--disable_lab_fk_dof", action="store_true",
                        help="Disable automatic opt_dof_pos override from sibling *_lab_fk.pkl.")
    parser.add_argument("--shadow_pkl_order", type=str, default="auto",
                        choices=["auto", "maniptrans_old", "urdf_xml"],
                        help="DOF order convention for shadow pkl.")

    args, _ = parser.parse_known_args()

    # ---- Load hand retarget pkls ----
    rh_data = None
    lh_data = None
    if args.rh_pkl and os.path.exists(args.rh_pkl):
        with open(args.rh_pkl, "rb") as f:
            rh_data = pickle.load(f)
        if not args.disable_lab_fk_dof:
            rh_data = maybe_override_with_lab_fk_dof(args.rh_pkl, rh_data, "RH")
        print(f"  RH: {rh_data['opt_wrist_pos'].shape[0]} frames, "
              f"{rh_data['opt_dof_pos'].shape[1]} DOFs")
        summarize_retarget_pkl(rh_data, "RH")

    if args.lh_pkl and os.path.exists(args.lh_pkl):
        with open(args.lh_pkl, "rb") as f:
            lh_data = pickle.load(f)
        if not args.disable_lab_fk_dof:
            lh_data = maybe_override_with_lab_fk_dof(args.lh_pkl, lh_data, "LH")
        print(f"  LH: {lh_data['opt_wrist_pos'].shape[0]} frames, "
              f"{lh_data['opt_dof_pos'].shape[1]} DOFs")
        summarize_retarget_pkl(lh_data, "LH")

    assert rh_data or lh_data, (
        "At least one hand pkl required. "
        f"RH path={args.rh_pkl} (exists={os.path.exists(args.rh_pkl) if args.rh_pkl else False}), "
        f"LH path={args.lh_pkl} (exists={os.path.exists(args.lh_pkl) if args.lh_pkl else False})"
    )

    obj_trajs_final = []
    obj_usd_list = []
    obj_usd_paths = [args.obj1_usd, args.obj2_usd, args.obj3_usd]
    T = 0

    if args.scene_pkl and os.path.exists(args.scene_pkl):
        # ===============================
        #  Mode A: Scene pkl (bimanual)
        # ===============================
        print(f"Loading scene: {args.scene_pkl}")
        with open(args.scene_pkl, "rb") as f:
            scene = pickle.load(f)

        obj_transf = scene["obj_transf"]
        obj_list = scene["obj_list"]
        mocap_frames = scene["mocap_frame_id_list"]

        # Frame range from program info
        frame_range = None
        vis_obj_ids = list(obj_list)

        if args.program_json and os.path.exists(args.program_json):
            with open(args.program_json) as f:
                _prog = json.load(f)
            prog = {eval(k): v for k, v in _prog.items()}

            stage_key = list(prog.keys())[args.stage]
            stage_info = prog[stage_key]
            lh_range, rh_range = stage_key[0], stage_key[1]

            if lh_range and rh_range:
                frame_range = (max(lh_range[0], rh_range[0]),
                               min(lh_range[1], rh_range[1]))
            elif rh_range:
                frame_range = tuple(rh_range)
            elif lh_range:
                frame_range = tuple(lh_range)

            # RH objects first, then LH
            vis_obj_ids = []
            for ol in [stage_info.get("obj_list_rh", []),
                        stage_info.get("obj_list_lh", [])]:
                for o in (ol or []):
                    if o in obj_transf and o not in vis_obj_ids:
                        vis_obj_ids.append(o)

            print(f"  Stage {args.stage}: range={frame_range}, "
                  f"primitive={stage_info.get('primitive')}")

        # Build frame_id_list
        step = args.subsample
        frame_id_list = mocap_frames[::step]
        if frame_range:
            frame_id_list = [f for f in frame_id_list
                             if frame_range[0] <= f <= frame_range[1]]
        T_display = len(frame_id_list)
        print(f"  MoCap: {len(mocap_frames)} frames, display: {T_display} (step={step})")

        # Extract object trajectories
        m2g = build_mujoco2gym()
        for obj_id in vis_obj_ids:
            if obj_id not in obj_transf:
                continue
            frames = [obj_transf[obj_id].get(fid, np.eye(4)) for fid in frame_id_list]
            raw = np.stack(frames, axis=0).astype(np.float32)
            traj = (m2g @ raw).astype(np.float32)
            obj_trajs_final.append(traj)
            print(f"  {obj_id}: {traj.shape[0]} frames, pos[0]={traj[0,:3,3]}")

        # Sync hand frames with object frames.
        # Retarget pkl covers only the stage range [fr0, fr1] at some stage_skip,
        # so hand_idx=0 corresponds to mocap fid=fr0, not fid=0. Recover stage_skip
        # from (stage_len / T_hand) and remap fid → hand index.
        T = T_display
        assert frame_range is not None, "program_json/frame_range required for hand-object sync"
        fr0, fr1 = frame_range
        stage_len = fr1 - fr0 + 1
        for label, data in [("rh", rh_data), ("lh", lh_data)]:
            if data is None:
                continue
            T_hand = data["opt_wrist_pos"].shape[0]
            stage_skip = stage_len / T_hand  # e.g. 2595/1297 ≈ 2.0
            hand_indices = np.clip(
                np.round([(fid - fr0) / stage_skip for fid in frame_id_list]).astype(int),
                0, T_hand - 1)
            if "opt_joints_pos" in data:
                data["opt_joints_pos"] = data["opt_joints_pos"][hand_indices]
            data["opt_wrist_pos"] = data["opt_wrist_pos"][hand_indices]
            data["opt_wrist_rot"] = data["opt_wrist_rot"][hand_indices]
            data["opt_dof_pos"] = data["opt_dof_pos"][hand_indices]
            print(f"  [{label}] Resampled: {T_hand} → {len(hand_indices)} frames "
                  f"(stage [{fr0},{fr1}], stage_skip≈{stage_skip:.2f})")

        # Match USD paths to vis_obj_ids
        for i in range(len(vis_obj_ids)):
            if i < len(obj_usd_paths) and obj_usd_paths[i]:
                obj_usd_list.append(obj_usd_paths[i])
            else:
                obj_usd_list.append(None)

    else:
        # ===============================
        #  Mode B: No scene pkl (GRAB demo style)
        # ===============================
        T_hand = min(
            rh_data["opt_wrist_pos"].shape[0] if rh_data else 999999,
            lh_data["opt_wrist_pos"].shape[0] if lh_data else 999999)
        T = T_hand

        # Load obj_traj pkl if provided
        if args.obj_traj and os.path.exists(args.obj_traj):
            with open(args.obj_traj, "rb") as f:
                otraj = pickle.load(f)
            opos = otraj["obj_pos"]       # (T, 3)
            orot_aa = otraj["obj_rot"]    # (T, 3) axis-angle
            orotmat = Rotation.from_rotvec(orot_aa).as_matrix()  # (T, 3, 3)
            T_obj = opos.shape[0]
            mat4x4 = np.zeros((T_obj, 4, 4), dtype=np.float32)
            mat4x4[:, :3, :3] = orotmat
            mat4x4[:, :3, 3] = opos
            mat4x4[:, 3, 3] = 1.0
            obj_trajs_final.append(mat4x4)
            T = min(T, T_obj)
            print(f"  ObjTraj: {args.obj_traj} ({T_obj} frames)")

        # Load obj2_traj pkl if provided
        if args.obj2_traj and os.path.exists(args.obj2_traj):
            with open(args.obj2_traj, "rb") as f:
                otraj2 = pickle.load(f)
            opos2 = otraj2["obj_pos"]
            orot_aa2 = otraj2["obj_rot"]
            orotmat2 = Rotation.from_rotvec(orot_aa2).as_matrix()
            T_obj2 = opos2.shape[0]
            mat4x4_2 = np.zeros((T_obj2, 4, 4), dtype=np.float32)
            mat4x4_2[:, :3, :3] = orotmat2
            mat4x4_2[:, :3, 3] = opos2
            mat4x4_2[:, 3, 3] = 1.0
            obj_trajs_final.append(mat4x4_2)
            T = min(T, T_obj2)
            print(f"  ObjTraj2: {args.obj2_traj} ({T_obj2} frames)")

        # USD for obj1, obj2
        if args.obj1_usd:
            obj_usd_list.append(args.obj1_usd)
        if args.obj2_traj and args.obj2_usd:
            obj_usd_list.append(args.obj2_usd)

    print(f"\n{'='*60}")
    print(f"  Retarget Viewer (Isaac Lab)")
    print(f"  Scene:   {args.scene_pkl or 'NONE (single-hand mode)'}")
    print(f"  RH:      {args.rh_pkl or 'NONE'}")
    print(f"  LH:      {args.lh_pkl or 'NONE'}")
    print(f"  ObjTraj: {args.obj_traj or 'from scene_pkl'}")
    print(f"  Frames:  {T}")
    print(f"{'='*60}")

    # ---- Resolve hand types ----
    DEXHAND_MAP = {
        "shadow":   ("shadow_right", "shadow_left"),
        "inspire":  ("inspire_right", None),
        "artimano": ("artimano_right", "artimano_left"),
        "allegro":  ("allegro_right", "allegro_left"),
    }
    rh_type, lh_type = DEXHAND_MAP[args.dexhand]
    # Disable LH if no lh_pkl or hand type doesn't support it
    if lh_data is None:
        lh_type = None

    # ---- Create viewer ----
    viewer = RetargetViewer(
        device=args.device,
        rh_type=rh_type,
        lh_type=lh_type,
        obj_usd_paths=obj_usd_list,
        lab_retarget=args.lab_retarget,
        shadow_pkl_order=args.shadow_pkl_order)

    # ---- Skeleton overlay ----
    retarget_skel = None
    retarget_bones = None
    mano_kp = None

    # Priority 1: retarget skeleton from opt_joints_pos in hand pkl
    if rh_data is not None and "opt_joints_pos" in rh_data:
        retarget_skel, retarget_bones = viewer.load_retarget_skeleton(rh_data, dexhand=args.dexhand)
        if retarget_skel.shape[0] > T:
            retarget_skel = retarget_skel[:T]
        print(f"  Retarget skeleton: ON ({retarget_skel.shape[0]} frames, {retarget_skel.shape[1]} bodies)")

    if retarget_skel is None:
        print(f"  Skeleton: OFF (no opt_joints_pos in pkl)")

    # ---- Play ----
    viewer.play(
        rh_data=rh_data,
        lh_data=lh_data,
        obj_trajs=obj_trajs_final,
        fps=args.fps,
        loop=not args.no_loop,
        speed=args.speed,
        retarget_skeleton=retarget_skel,
        retarget_bones=retarget_bones,
        physics_playback=args.physics_playback)


if __name__ == "__main__":
    main()
