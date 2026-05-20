#!/usr/bin/env python3
"""
Retarget Gym/Lab sv_dict -> Isaac-Lab DOFs by fitting to opt_joints_pos.

Why:
- Existing retarget pkl (e.g. 102_sv_dict.pkl) may not overlay with current
  Isaac-Lab USD due to model/axis differences.
- This script recomputes opt_dof_pos for the *current Lab URDF* while
  keeping the same wrist trajectory and target joint positions.

Input:
  - pkl with keys: opt_wrist_pos [T,3], opt_wrist_rot [T,3], opt_joints_pos [T,N,3]
Output:
  - pkl with updated opt_dof_pos [T,D] (URDF XML order), plus metadata fields.
"""

from __future__ import annotations

import argparse
import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np
import pytorch_kinematics as pk
import torch


HAND_BODY_NAMES = {
    "shadow": [
        "palm",
        "ffknuckle", "ffproximal", "ffmiddle", "ffdistal", "fftip",
        "lfmetacarpal", "lfknuckle", "lfproximal", "lfmiddle", "lfdistal", "lftip",
        "mfknuckle", "mfproximal", "mfmiddle", "mfdistal", "mftip",
        "rfknuckle", "rfproximal", "rfmiddle", "rfdistal", "rftip",
        "thbase", "thproximal", "thhub", "thmiddle", "thdistal", "thtip",
    ],
    "inspire": [
        "hand_base_link",
        "index_proximal", "index_intermediate", "index_tip",
        "middle_proximal", "middle_intermediate", "middle_tip",
        "pinky_proximal", "pinky_intermediate", "pinky_tip",
        "ring_proximal", "ring_intermediate", "ring_tip",
        "thumb_proximal_base", "thumb_proximal", "thumb_intermediate", "thumb_distal", "thumb_tip",
    ],
    "artimano": [
        "palm",
        "index1y", "index1z", "index2", "index3", "index_tip",
        "middle1y", "middle1z", "middle2", "middle3", "middle_tip",
        "pinky1y", "pinky1z", "pinky2", "pinky3", "pinky_tip",
        "ring1y", "ring1z", "ring2", "ring3", "ring_tip",
        "thumb1x", "thumb1y", "thumb1z", "thumb2y", "thumb2z", "thumb3", "thumb_tip",
    ],
    "allegro": [
        "base_link",
        "link_0", "link_4", "link_8", "link_12",
        "link_1", "link_5", "link_9", "link_13",
        "link_2", "link_6", "link_10", "link_14",
        "link_3", "link_7", "link_11", "link_15",
        "link_3_tip", "link_7_tip", "link_11_tip", "link_15_tip",
    ],
}

TIP_IDS = {
    "shadow": [5, 11, 16, 21, 27],
    "inspire": [3, 6, 9, 12, 17],
    "artimano": [5, 10, 15, 20, 27],
    "allegro": [17, 18, 19, 20],
}

DISTAL_IDS = {
    "shadow": [4, 10, 15, 20, 26],
    "inspire": [2, 5, 8, 11, 16],
    "artimano": [4, 9, 14, 19, 26],
    "allegro": [13, 14, 15, 16],
}

SIDE_PREFIX = {
    "shadow": {"right": "r_", "left": "l_"},
    "inspire": {"right": "R_", "left": "L_"},
    "artimano": {"right": "r_", "left": "l_"},
    "allegro": {"right": "r_", "left": "l_"},
}

DEFAULT_URDF = {
    ("shadow", "right"): "/home/david/david/Again_0420/data/assets/shadow_hand/shadow_hand_woarm_right.urdf",
    ("shadow", "left"): "/home/david/david/Again_0420/data/assets/shadow_hand/shadow_hand_woarm_left.urdf",
    ("inspire", "right"): "/home/david/david/Again_0420/data/assets/inspire_hand/inspire_hand_right.urdf",
    ("inspire", "left"): "/home/david/david/Again_0420/data/assets/inspire_hand/inspire_hand_left.urdf",
    ("artimano", "right"): "/home/david/david/Again_0420/data/assets/mano/rh_mano_lab.urdf",
    ("artimano", "left"): "/home/david/david/Again_0420/data/assets/mano/lh_mano.urdf",
    ("allegro", "right"): "/home/david/david/Again_0420/data/assets/allegro_hand/allegro_hand_right.urdf",
    ("allegro", "left"): "/home/david/david/Again_0420/data/assets/allegro_hand/allegro_hand_left.urdf",
}


def parse_xml_dof_order(urdf_path: str) -> list[str]:
    root = ET.parse(urdf_path).getroot()
    return [
        j.attrib["name"]
        for j in root.findall("joint")
        if j.attrib.get("type", "") in ("revolute", "prismatic", "continuous")
    ]


def parse_joint_limits_xml(urdf_path: str, dof_names_xml: list[str]) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    jmap = {}
    for j in root.findall("joint"):
        n = j.attrib.get("name", "")
        if j.attrib.get("type", "") not in ("revolute", "prismatic", "continuous"):
            continue
        lim = j.find("limit")
        lo = float(lim.attrib["lower"]) if lim is not None and "lower" in lim.attrib else -np.pi
        hi = float(lim.attrib["upper"]) if lim is not None and "upper" in lim.attrib else np.pi
        jmap[n] = (lo, hi)
    lo = np.array([jmap[n][0] for n in dof_names_xml], dtype=np.float32)
    hi = np.array([jmap[n][1] for n in dof_names_xml], dtype=np.float32)
    return lo, hi


def canonical_name(name: str) -> str:
    n = name.strip()
    for p in ("r_", "l_", "R_", "L_"):
        if n.startswith(p):
            n = n[len(p):]
            break
    return n.replace(".", "_").lower()


def map_names_by_canonical(target_names: list[str], source_names: list[str]) -> list[int] | None:
    idx = []
    source_lookup = {}
    for i, n in enumerate(source_names):
        source_lookup.setdefault(canonical_name(n), i)
    for t in target_names:
        c = canonical_name(t)
        if c not in source_lookup:
            return None
        idx.append(source_lookup[c])
    return idx


def resolve_body_names_for_chain(body_names: list[str], chain_link_names: list[str], prefix: str) -> list[str]:
    link_set = set(chain_link_names)
    resolved = []
    for name in body_names:
        candidates = [name]
        raw = name
        for p in ("r_", "l_", "R_", "L_"):
            if raw.startswith(p):
                raw = raw[len(p):]
                break
        candidates.append(raw)
        candidates.append(prefix + raw)
        candidates.append(name.replace(".", "_"))
        candidates.append(name.replace("_", "."))
        candidates.append((prefix + raw).replace(".", "_"))
        candidates.append((prefix + raw).replace("_", "."))
        chosen = None
        for c in candidates:
            if c in link_set:
                chosen = c
                break
        if chosen is None:
            raise ValueError(
                f"Body link '{name}' is not found in URDF chain. "
                f"First links: {chain_link_names[:10]}"
            )
        resolved.append(chosen)
    return resolved


def aa_to_rotmat_torch(aa: torch.Tensor) -> torch.Tensor:
    # aa: [3]
    angle = torch.norm(aa) + 1e-12
    axis = aa / angle
    x, y, z = axis[0], axis[1], axis[2]
    K = torch.tensor(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=aa.dtype,
        device=aa.device,
    )
    I = torch.eye(3, dtype=aa.dtype, device=aa.device)
    return I + torch.sin(angle) * K + (1.0 - torch.cos(angle)) * (K @ K)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_pkl", type=str, required=True)
    parser.add_argument("--out_pkl", type=str, required=True)
    parser.add_argument(
        "--dexhand",
        type=str,
        default="shadow",
        choices=["shadow", "inspire", "artimano", "allegro"],
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
    )
    parser.add_argument("--side", type=str, choices=["right", "left"], default="right")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--iters", type=int, default=250)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--smooth_w", type=float, default=2e-3)
    parser.add_argument("--tip_w", type=float, default=4.0)
    parser.add_argument("--distal_w", type=float, default=2.0)
    args = parser.parse_args()

    if args.urdf is None:
        args.urdf = DEFAULT_URDF[(args.dexhand, args.side)]

    assert os.path.exists(args.in_pkl), f"Missing input pkl: {args.in_pkl}"
    assert os.path.exists(args.urdf), f"Missing urdf: {args.urdf}"

    with open(args.in_pkl, "rb") as f:
        src = pickle.load(f)

    for k in ("opt_wrist_pos", "opt_wrist_rot", "opt_joints_pos"):
        assert k in src, f"Input pkl missing key: {k}"

    wrist_pos_np = np.asarray(src["opt_wrist_pos"], dtype=np.float32)
    wrist_rot_np = np.asarray(src["opt_wrist_rot"], dtype=np.float32)
    joints_np = np.asarray(src["opt_joints_pos"], dtype=np.float32)
    T = wrist_pos_np.shape[0]
    assert joints_np.shape[0] == T, "Frame count mismatch"

    side_prefix = SIDE_PREFIX[args.dexhand][args.side]

    # Build chain and DOF orders.
    with open(args.urdf, "r", encoding="utf-8", errors="ignore") as f:
        urdf_text = f.read()
    if "?>" in urdf_text:
        urdf_text = urdf_text.split("?>", 1)[1]
    chain = pk.build_chain_from_urdf(urdf_text).to(dtype=torch.float32, device=args.device)
    dof_chain = list(chain.get_joint_parameter_names())
    dof_xml = parse_xml_dof_order(args.urdf)
    assert len(dof_chain) == len(dof_xml), f"DOF count mismatch: chain={len(dof_chain)} xml={len(dof_xml)}"
    print(f"[info] dexhand={args.dexhand} side={args.side}")
    print(f"[info] urdf={args.urdf}")
    print(f"[info] dof_count={len(dof_xml)}")
    print(f"[info] body_count(target joints)={joints_np.shape[1]}")

    xml2chain = [dof_chain.index(n) for n in dof_xml]
    chain2xml = [dof_xml.index(n) for n in dof_chain]

    if isinstance(src.get("meta_body_names"), (list, tuple)) and len(src["meta_body_names"]) > 0:
        body_names_req = list(src["meta_body_names"])
    else:
        body_names_req = [side_prefix + n for n in HAND_BODY_NAMES[args.dexhand]]
    body_names = resolve_body_names_for_chain(body_names_req, list(chain.get_link_names()), side_prefix)
    assert len(body_names) == joints_np.shape[1], (
        f"Body count mismatch: body_names={len(body_names)} vs opt_joints_pos={joints_np.shape[1]}"
    )

    lo_xml, hi_xml = parse_joint_limits_xml(args.urdf, dof_xml)
    lo_chain = torch.as_tensor(lo_xml[xml2chain], device=args.device)
    hi_chain = torch.as_tensor(hi_xml[xml2chain], device=args.device)

    q_out_xml = np.zeros((T, len(dof_xml)), dtype=np.float32)
    per_frame_err = []
    prev_q_chain = None

    # Warm-start from existing dof if compatible.
    if "opt_dof_pos" in src:
        q_src = np.asarray(src["opt_dof_pos"], dtype=np.float32)
        q_src_xml = None
        meta_joint_names = src.get("meta_joint_names", None)
        if isinstance(meta_joint_names, (list, tuple)) and len(meta_joint_names) == q_src.shape[1]:
            gather = map_names_by_canonical(dof_xml, list(meta_joint_names))
            if gather is not None:
                q_src_xml = q_src[:, gather]
        if q_src_xml is None and q_src.shape == (T, len(dof_xml)):
            q_src_xml = q_src
        if q_src_xml is not None and q_src_xml.shape == (T, len(dof_xml)):
            prev_q_chain = torch.as_tensor(q_src_xml[0, xml2chain], device=args.device)

    # Joint weights.
    n_bodies = joints_np.shape[1]
    w = torch.ones((n_bodies,), dtype=torch.float32, device=args.device)
    for i in TIP_IDS.get(args.dexhand, []):
        if i < n_bodies:
            w[i] = float(args.tip_w)
    for i in DISTAL_IDS.get(args.dexhand, []):
        if i < n_bodies:
            w[i] = float(args.distal_w)
    w = w / w.mean()

    for t in range(T):
        target = torch.as_tensor(joints_np[t], device=args.device)
        wrist_pos = torch.as_tensor(wrist_pos_np[t], device=args.device)
        wrist_rot = torch.as_tensor(wrist_rot_np[t], device=args.device)
        Rw = aa_to_rotmat_torch(wrist_rot)  # [3,3]

        if prev_q_chain is None:
            q0 = torch.zeros((len(dof_chain),), dtype=torch.float32, device=args.device)
        else:
            q0 = prev_q_chain.detach().clone()

        q_var = q0.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([q_var], lr=args.lr)

        for _ in range(args.iters):
            q = torch.clamp(q_var, lo_chain, hi_chain)
            fk = chain.forward_kinematics(q.unsqueeze(0))
            loc = torch.stack([fk[n].get_matrix()[0, :3, 3] for n in body_names], dim=0)  # [28,3]
            world = (Rw @ loc.T).T + wrist_pos

            diff = world - target
            loss_pos = torch.mean(torch.norm(diff, dim=-1) * w)
            loss = loss_pos
            if prev_q_chain is not None:
                loss = loss + float(args.smooth_w) * torch.mean((q - prev_q_chain) ** 2)

            opt.zero_grad()
            loss.backward()
            opt.step()

        q_final = torch.clamp(q_var.detach(), lo_chain, hi_chain)
        prev_q_chain = q_final

        # Final fitting error.
        with torch.no_grad():
            fk = chain.forward_kinematics(q_final.unsqueeze(0))
            loc = torch.stack([fk[n].get_matrix()[0, :3, 3] for n in body_names], dim=0)
            world = (Rw @ loc.T).T + wrist_pos
            frame_err = torch.norm(world - target, dim=-1).mean().item()
            per_frame_err.append(frame_err)

        q_xml = q_final[chain2xml].cpu().numpy()
        q_out_xml[t] = q_xml

        if t % 10 == 0 or t == T - 1:
            print(f"[{t:03d}/{T}] mean joint err = {per_frame_err[-1]:.5f} m")

    out = dict(src)
    out["opt_dof_pos"] = q_out_xml
    out["_retargeted_against"] = f"isaac_lab_{args.dexhand}_urdf_fit_from_opt_joints"
    out["_retargeted_dexhand"] = args.dexhand
    out["_retargeted_side"] = args.side
    out["_retargeted_urdf"] = args.urdf
    out["_fit_mean_joint_err"] = float(np.mean(per_frame_err))
    out["_fit_max_joint_err"] = float(np.max(per_frame_err))
    out["_fit_per_frame_mean_joint_err"] = np.asarray(per_frame_err, dtype=np.float32)

    os.makedirs(os.path.dirname(args.out_pkl), exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(out, f)

    print("\nDone")
    print(f"  out_pkl: {args.out_pkl}")
    print(f"  mean_err: {out['_fit_mean_joint_err']:.6f} m")
    print(f"  max_err : {out['_fit_max_joint_err']:.6f} m")


if __name__ == "__main__":
    main()
