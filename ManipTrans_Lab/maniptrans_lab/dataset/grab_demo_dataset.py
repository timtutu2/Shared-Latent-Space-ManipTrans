"""
Lightweight dataset loader for the grab_demo / motion_data pkls that ship
in this repo's `data/` directory (/home/david/david/Again_0420/data).

Unlike ManipTrans's original `GrabDatasetDexhand` — which needs chamfer
distance, pytorch3d, SMPLX, and re-runs MANO→dex retargeting — this loader
just reads the pre-extracted pkl files:

    data/motion_data/grab_demo/{seq}/{seq}_mano_task.pkl
        mano_joints_pos   [T, 16, 3]   (world-frame, quat/aa mix)
        mano_tips_pos     [T,  5, 3]
        wrist_pos         [T, 3]
        wrist_rot         [T, 4]  (quat wxyz)
        obj_pos           [T, 3]
        obj_rot           [T, 4]  (quat wxyz)
        meta_joint_names  list[str]
        meta_tip_names    list[str]
        meta_table_surface_z  float
        fps               float

    data/motion_data/grab_demo/retargeting/mano2{embod}_{side}/{seq}_sv_dict.pkl
        opt_dof_pos       [T, n_dofs]
        opt_joints_pos    [T, n_bodies, 3]
        opt_wrist_pos     [T, 3]
        opt_wrist_rot     [T, 3]   (axis-angle)

    data/motion_data/grab_demo/object/{seq}_obj_traj.pkl
        obj_pos           [T, 3]
        obj_rot           [T, 3]   (axis-angle)

Index convention: `"g102"` -> seq "102".

Output matches the dict that DexImitatorEnv._pack_data consumes.
"""
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.decorators import register_manipdata
from dataset.transform import aa_to_rotmat, quat_to_rotmat, rotmat_to_aa


DEFAULT_DATA_ROOT = os.environ.get(
    "MANIPTRANS_DATA_ROOT",
    "/home/david/david/Again_0420/data",
)


def _finite_diff_vel(p: torch.Tensor, dt: float) -> torch.Tensor:
    v = torch.zeros_like(p)
    v[:-1] = (p[1:] - p[:-1]) / dt
    v[-1] = v[-2] if p.shape[0] >= 2 else 0.0
    return v


def _finite_diff_ang_vel(R: torch.Tensor, dt: float) -> torch.Tensor:
    """R: [T, 3, 3] -> [T, 3] (axis-angle per-frame angular velocity)."""
    T = R.shape[0]
    diff = R[1:] @ R[:-1].transpose(-1, -2)                  # [T-1, 3, 3]
    aa = rotmat_to_aa(diff) / dt                             # [T-1, 3]
    aa = torch.cat([aa, aa[-1:]], dim=0) if T > 1 else torch.zeros(T, 3, device=R.device)
    return aa


def _tips_distance(
    mano_joints: dict[str, torch.Tensor],
    obj_rotmat: torch.Tensor,
    obj_pos: torch.Tensor,
    obj_verts: torch.Tensor,
) -> torch.Tensor:
    """Compute per-frame nearest fingertip distance to object mesh points: [T, 5]."""
    tip_names = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    any_tip = next(iter(mano_joints.values()))
    T = any_tip.shape[0]
    device = any_tip.device

    # Transform object vertices to world frame per timestep.
    # obj_world: [T, V, 3]
    obj_world = torch.einsum("tij,vj->tvi", obj_rotmat, obj_verts) + obj_pos[:, None, :]

    dists = []
    for name in tip_names:
        if name in mano_joints:
            tip = mano_joints[name][:, None, :]  # [T, 1, 3]
            # [T, 1, V] -> [T]
            near = torch.cdist(tip, obj_world).amin(dim=-1).squeeze(1)
        else:
            # Missing tip mapping: keep neutral value (outside contact band).
            near = torch.full((T,), 0.03, device=device, dtype=torch.float32)
        dists.append(near)
    return torch.stack(dists, dim=1)


class _GrabDemoBase(Dataset):
    """
    Concrete ManipDataFactory-registered loader shared by RH and LH variants.

    Required kwargs (passed in by DexImitatorEnv._load_demo_data):
        device, dexhand, mujoco2gym_transf (4x4 torch), max_seq_len, embodiment
    Optional:
        data_root  (defaults to $MANIPTRANS_DATA_ROOT)
    """

    SIDE = "rh"  # subclasses override

    def __init__(
        self,
        *,
        device,
        dexhand,
        mujoco2gym_transf,
        max_seq_len,
        embodiment,
        data_root=None,
        skip: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.dexhand = dexhand
        self.mujoco2gym_transf = mujoco2gym_transf
        self.max_seq_len = int(max_seq_len)
        self.embodiment = embodiment
        self.skip = skip
        # grab_demo/object/*_obj_traj.pkl in this workspace is already in the
        # Isaac/retarget frame. Applying mujoco2gym again shifts object pose.
        self.apply_mujoco2gym_to_obj = bool(kwargs.get("apply_mujoco2gym_to_obj", False))

        self.data_root = data_root or DEFAULT_DATA_ROOT
        self.grab_root = os.path.join(self.data_root, "motion_data", "grab_demo")

        # sequence discovery: every numeric subdir of grab_demo/ is a sequence
        seqs = []
        if os.path.isdir(self.grab_root):
            for name in sorted(os.listdir(self.grab_root)):
                p = os.path.join(self.grab_root, name)
                if os.path.isdir(p) and name.isdigit():
                    seqs.append(name)
        self._seqs = seqs

    # -- Dataset API --------------------------------------------------- #
    def __len__(self):
        return len(self._seqs)

    def _idx_to_seq(self, idx):
        if isinstance(idx, str):
            seq = idx
            if seq.startswith("g"):
                seq = seq[1:]
        else:
            seq = str(idx)
        return seq

    def __getitem__(self, idx):
        seq = self._idx_to_seq(idx)
        device = self.device

        mano_task_fp = os.path.join(self.grab_root, seq, f"{seq}_mano_task.pkl")
        sv_fp = os.path.join(self.grab_root, "retargeting",
                             f"mano2{self.embodiment}_{self.SIDE}", f"{seq}_sv_dict.pkl")
        obj_fp = os.path.join(self.grab_root, "object", f"{seq}_obj_traj.pkl")

        mano = self._load(mano_task_fp)
        sv = self._load(sv_fp)
        obj = self._load(obj_fp) if os.path.exists(obj_fp) else None

        fps = float(mano.get("fps", 30.0))
        dt = 1.0 / (fps / max(1, self.skip))

        # --- retargeted dexhand trajectory (primary wrist / joint source) ----
        # Keep the same wrist anchor convention as view_retarget.py:
        # use retargeted palm/body-0 position when available.
        opt_joints = torch.as_tensor(sv["opt_joints_pos"], device=device, dtype=torch.float32)
        if opt_joints.ndim == 3 and opt_joints.shape[1] > 0:
            wrist_pos = opt_joints[:, 0, :].clone()
        else:
            wrist_pos = torch.as_tensor(sv["opt_wrist_pos"], device=device, dtype=torch.float32)
        wrist_rot_aa = torch.as_tensor(sv["opt_wrist_rot"], device=device, dtype=torch.float32)
        opt_dof_pos = torch.as_tensor(sv["opt_dof_pos"], device=device, dtype=torch.float32)

        # === Inspire pkl → common reorder (user-specified target sim values) ===
        # After this + engine identity-by-name → sim, each sim slot receives
        # exactly what the user specified:
        #   sim[ 0] R_thumb_proximal_yaw_joint     ← pkl[ 8] = 1.01060
        #   sim[ 1] R_thumb_proximal_pitch_joint   ← pkl[ 3] = 0.08444
        #   sim[ 2] R_index_proximal_joint         ← pkl[ 0] = 0.16590
        #   sim[ 3] R_middle_proximal_joint        ← pkl[ 2] = 0.66931
        #   sim[ 4] R_pinky_proximal_joint         ← pkl[ 4] = 0.00000
        #   sim[ 5] R_ring_proximal_joint          ← pkl[ 6] = 0.22573
        #   sim[ 6] R_thumb_intermediate_joint     ← pkl[ 9] = 0.30959
        #   sim[ 7] R_index_intermediate_joint     ← pkl[ 1] = 0.29376
        #   sim[ 8] R_middle_intermediate_joint    ← pkl[11] = 0.02115
        #   sim[ 9] R_pinky_intermediate_joint     ← pkl[ 5] = 1.68067
        #   sim[10] R_ring_intermediate_joint      ← pkl[ 7] = 1.22685
        #   sim[11] R_thumb_distal_joint           ← pkl[10] = 0.19224
        # Via common (InspireRH.dof_names) indexing:
        #   common[i] ← pkl[perm[i]]  where perm =
        if self.embodiment == "inspire":
            perm = [0, 1, 2, 11, 4, 5, 6, 7, 8, 3, 9, 10]
            opt_dof_pos = opt_dof_pos[:, perm].clone()

        T = wrist_pos.shape[0]
        if T > self.max_seq_len:
            T = self.max_seq_len
            wrist_pos = wrist_pos[:T]
            wrist_rot_aa = wrist_rot_aa[:T]
            opt_joints = opt_joints[:T]
            opt_dof_pos = opt_dof_pos[:T]

        # The retargeted sv_dict (opt_wrist_pos / opt_wrist_rot / opt_joints_pos)
        # was produced by running ManipTrans's mano2dexhand.py inside Isaac Gym,
        # so these tensors are ALREADY in the sim frame (no mujoco→gym transform
        # needed). BUT Isaac Lab's USD import bakes an Rx(+90°) rotation into the
        # root compared to the original URDF convention (URDF fingers along -Y,
        # USD fingers along world -Z at identity — see dexhands/inspire.py and
        # tools/inspect_usd_identity.py). Q_usd = R_urdf @ Rx(-90°).
        wrist_rot_mat = aa_to_rotmat(wrist_rot_aa)
        if hasattr(self.dexhand, "usd_identity_rotmat_inv"):
            inv = torch.tensor(self.dexhand.usd_identity_rotmat_inv,
                               device=device, dtype=torch.float32)
            wrist_rot_mat = wrist_rot_mat @ inv
            wrist_rot_aa = rotmat_to_aa(wrist_rot_mat)

        # velocities from the retargeted trajectory
        wrist_vel = _finite_diff_vel(wrist_pos, dt)
        wrist_ang_vel = _finite_diff_ang_vel(wrist_rot_mat, dt)

        # --- build mano_joints dict keyed by MANO joint name ----------------
        # Use the retargeted per-body positions (opt_joints) indexed by the
        # dex body ordering, but surfaced under MANO names so the env's
        # _pack_data can look them up via dexhand.to_hand(dex_body)[0].
        body_names = self.dexhand.body_names          # 18 entries for inspire
        mano_joints = {}
        for body_idx, dex_body in enumerate(body_names):
            mano_list = self.dexhand.to_hand(dex_body)
            if not mano_list:
                continue
            mano_name = mano_list[0]
            if mano_name == "wrist":
                continue
            # last-write wins if multiple dex bodies share a MANO name
            mano_joints[mano_name] = opt_joints[:, body_idx]

        mano_joints_velocity = {k: _finite_diff_vel(v, dt) for k, v in mano_joints.items()}

        # --- object trajectory [T, 4, 4] -----------------------------------
        if obj is not None:
            obj_pos_np = np.asarray(obj["obj_pos"])[:T]
            obj_rot_np = np.asarray(obj["obj_rot"])[:T]
            obj_pos = torch.as_tensor(obj_pos_np, device=device, dtype=torch.float32)
            obj_rot_aa = torch.as_tensor(obj_rot_np, device=device, dtype=torch.float32)
            obj_rotmat = aa_to_rotmat(obj_rot_aa)
            obj_traj = torch.eye(4, device=device).unsqueeze(0).repeat(T, 1, 1)
            obj_traj[:, :3, :3] = obj_rotmat
            obj_traj[:, :3, 3] = obj_pos
            if self.apply_mujoco2gym_to_obj:
                obj_traj = self.mujoco2gym_transf @ obj_traj
        else:
            # imitator tasks don't need a real obj trajectory, but the env expects the key.
            # Synthesize an identity trajectory parked at the origin.
            obj_traj = torch.eye(4, device=device).unsqueeze(0).repeat(T, 1, 1)
            obj_pos = obj_traj[:, :3, 3]
            obj_rotmat = obj_traj[:, :3, :3]

        # obj vertices (for BPS encoding in full imitator); optional
        obj_verts = self._load_obj_verts(seq, T)
        # Dynamic targets used by manipulation-stage reward/reset.
        obj_pos_world = obj_traj[:, :3, 3]
        obj_rot_world = obj_traj[:, :3, :3]
        obj_vel = _finite_diff_vel(obj_pos_world, dt)
        obj_ang_vel = _finite_diff_ang_vel(obj_rot_world, dt)
        tip_dist = _tips_distance(mano_joints, obj_rot_world, obj_pos_world, obj_verts)

        return {
            "obj_trajectory": obj_traj,
            "obj_velocity": obj_vel,
            "obj_angular_velocity": obj_ang_vel,
            "tips_distance": tip_dist,
            "wrist_pos": wrist_pos,
            "wrist_rot": wrist_rot_aa,
            "wrist_velocity": wrist_vel,
            "wrist_angular_velocity": wrist_ang_vel,
            "mano_joints": mano_joints,
            "mano_joints_velocity": mano_joints_velocity,
            "obj_verts": obj_verts,
            # extra fields consumers may use
            "opt_dof_pos": opt_dof_pos,
            "fps": fps,
        }

    # -- helpers ------------------------------------------------------ #
    @staticmethod
    def _load(path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[grab_demo_dataset] required pkl not found: {path}. "
                f"Check $MANIPTRANS_DATA_ROOT and the seq index.")
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_obj_verts(self, seq, T):
        """Load the object mesh vertices; fallback to a small random cube."""
        obj_path = os.path.join(self.grab_root, seq, f"{seq}_obj.obj")
        if os.path.exists(obj_path):
            verts = []
            with open(obj_path, "r") as f:
                for line in f:
                    if line.startswith("v "):
                        parts = line.split()
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if verts:
                v = torch.tensor(verts, device=self.device, dtype=torch.float32)
                # subsample to 1024 points for a compact BPS footprint
                if v.shape[0] > 1024:
                    idx = torch.randperm(v.shape[0], device=self.device)[:1024]
                    v = v[idx]
                return v
        # fallback: 64 points on a small cube
        return (torch.rand(64, 3, device=self.device) - 0.5) * 0.1


@register_manipdata("grabdemo_rh")
class GrabDemoDexHandRH(_GrabDemoBase):
    SIDE = "rh"


@register_manipdata("grabdemo_lh")
class GrabDemoDexHandLH(_GrabDemoBase):
    SIDE = "lh"
