# ManipTrans_Lab – Data & First Run Walkthrough

This guide assumes you already have:

* `~/miniforge3/envs/maniptrans_lab` — the conda env with PyTorch, Isaac Lab,
  and the packages in `requirements.txt` installed.
* The data layout under `/home/david/david/Again_0420/data/`:
  * `data/assets/` – Isaac-Lab-ready USDs (inspire, allegro, shadow, mano…)
  * `data/motion_data/` – per-sequence pkl files (grab_demo, 083f7a_demo, 0f020d_demo)

## What's already wired for you

The InspireRH dexhand now points at the Isaac Lab USD
`data/assets/inspire_hand/inspire_hand_right_lab_0404/inspire_hand_right_lab.usd`,
and a new **grab_demo** dataset loader
([maniptrans_lab/dataset/grab_demo_dataset.py](maniptrans_lab/dataset/grab_demo_dataset.py))
reads the pkl schema that ships in `data/motion_data/grab_demo/`:

| pkl                                                                | used for                                         |
| ------------------------------------------------------------------ | ------------------------------------------------ |
| `grab_demo/<seq>/<seq>_mano_task.pkl`                              | raw MANO wrist/joint poses, object poses, fps    |
| `grab_demo/retargeting/mano2<embodiment>_<side>/<seq>_sv_dict.pkl` | retargeted dexhand DoFs + wrist (primary source) |
| `grab_demo/object/<seq>_obj_traj.pkl`                              | manipulated object trajectory                    |
| `grab_demo/<seq>/<seq>_obj.obj`                                    | object mesh (for BPS/contact reward)             |

The default index `"g102"` in
[data/envs/dex_imitator_inspire_rh_env.yaml](data/envs/dex_imitator_inspire_rh_env.yaml)
routes to `grab_demo/102/`.

## Step 1 — verify the data loader (no Isaac Lab needed)

This is the cheapest sanity check: it exercises the factory, the dexhand
wrapper, the pkl reader, the MANO↔dex body mapping, and the frame transforms.

```bash
cd /home/david/david/Again_0420/ManipTrans_Lab
conda activate maniptrans_lab
python tools/smoke_test_dataset.py
```

**Expected output (last ~20 lines):**

```
[OK] device = cuda:0
[OK] dexhand = inspire_rh (12 dofs, 18 bodies)
[OK] dataset_type('g102') = grabdemo
[OK] created GrabDemoDexHandRH with 1 seqs: ['102']
[OK] sequence g102 loaded:
    obj_trajectory            [60, 4, 4] torch.float32
    wrist_pos                 [60, 3] torch.float32
    wrist_rot                 [60, 3] torch.float32
    wrist_velocity            [60, 3] torch.float32
    wrist_angular_velocity    [60, 3] torch.float32
    mano_joints (dict with 16 keys: ['index_intermediate', ...])
    mano_joints_velocity (dict with 16 keys: ...)
    obj_verts                 [1024, 3] torch.float32
    opt_dof_pos               [60, 12] torch.float32
    fps                       30.0
[OK] mano_joints has 16 keyed entries — env's _pack_data will index these.
[OK] every non-wrist dex body has a matching MANO entry.

SMOKE TEST PASSED ✓
```

If this fails with `UserWarning: [ManipDataFactory] skipped dataset.grab_dataset_dexhand`,
that is fine — it just means the heavy GRAB loader needs `manotorch / pytorch3d /
chamfer_distance`, which the lightweight `grab_demo_dataset` does not.

## Step 2 — verify Isaac Lab can load the dexhand USD

**Run from the `ManipTrans_Lab/` project root** (not from `maniptrans_lab/`) —
the arg-file's paths (`data/envs/...`, `args/...`) are relative to the
project root, matching MimicKit's convention.

This runs `run.py` in **test mode** with `num_envs=1` and `visualize=True`.
It opens the Isaac Sim viewer, spawns an Inspire right-hand at the default
pose, and plays the demo policy (untrained, so the hand just twitches
around the demo's start pose).

```bash
cd /home/david/david/Again_0420/ManipTrans_Lab    # ← project root, NOT maniptrans_lab/
python maniptrans_lab/run.py \
    --arg_file args/dex_imitator_inspire_rh_ppo_args.txt \
    --num_envs 1 \
    --mode test \
    --visualize True \
    --test_episodes 2
```

**Actual observed output (verified on this workstation, headless):**

```
Setting seed: 454847609887
Overriding Engine configs with parameters from the Environment:
    control_mode: pos
    control_freq: 60
    sim_freq: 120

Building dex_imitator env
[INFO][AppLauncher]: Using device: cuda:0
[INFO][AppLauncher]: Loading experience file: .../isaaclab.python.headless.kit
[INFO][IsaacLab]: Logging to file: /tmp/isaaclab/logs/isaaclab_*.log

Building 1/1 envs
Building Obj:0 in 1/1 envs
Initializing simulation...
loading demo sequences: 100%|██████████| 1/1 [00:00<00:00]
Building DictPPO agent
Total parameter count: 2255892
Mean Return: 0.10017707943916321
Mean Episode Length: 11.0
Episodes: 1
```

A viewer window should pop up (when `--visualize True` and `$DISPLAY` is set)
showing an Inspire right hand at the demo's start pose. Without a display,
just the log output above is enough to confirm the port works. Mean Episode
Length < 60 is expected for an untrained policy — the wrist-error FAIL
check terminates each rollout as soon as the hand drifts > 30 cm from the
demo wrist target.

### Harmless warnings you will see

These are expected and do not indicate a problem:

* `[ManipDataFactory] skipped dataset.grab_dataset_dexhand: No module named 'manotorch'`
  — the heavy original GRAB loader isn't installed; the lightweight
  `grab_demo_dataset` takes over.
* `[omni.usd] Warning: ... Unresolved reference prim path ...R_thumb_tip/visuals ...`
  — the lab USD references visuals from `inspire_hand_right_lab_physics.usd`
  which don't exist in the shipped asset; physics work but fingertip visual
  meshes may not render. (If you need visuals, re-export the USD from the
  URDF with visuals embedded.)
* `[omni.physx.plugin] PhysicsUSD: CreateJoint - found a joint with disjointed
  body transforms` — Isaac Lab auto-snaps joint transforms; fine for
  manipulation tasks.
* `[gpu.foundation.plugin] CUDA peer-to-peer ...` — P2P bandwidth probe on
  multi-GPU systems, informational.

### If the viewer doesn't open

* `--visualize True` is required; by default it's `True` in run.py but the
  engine forces headless if `DISPLAY` is unset. Export `DISPLAY=:0` (or
  equivalent) or stream over VNC/x11 forwarding.
* If the env reports `No such file or directory: data/assets/inspire_hand/...`
  — the relative resolver lands at `/home/david/david/Again_0420/`. Either
  leave the project at that location, set `MANIPTRANS_DATA_ROOT`, or edit
  `inspire.py:_urdf_path` to an absolute path.

## Step 3 — short training run (sanity-check RL loop)

```bash
cd /home/david/david/Again_0420/ManipTrans_Lab    # ← project root, NOT maniptrans_lab/
python maniptrans_lab/run.py \
    --arg_file args/dex_imitator_inspire_rh_ppo_args.txt \
    --num_envs 256 \
    --mode train \
    --visualize False \
    --max_samples 200000
```

Runs ~200k samples (≈ a couple minutes on a single RTX-class GPU). Watch
the log for:

```
Iteration    10  Samples    3200  Mean Return  0.15  ...
Iteration    50  Samples   16000  Mean Return  0.32  ...
Iteration   100  Samples   32000  Mean Return  0.48  ...
```

Mean return should trend upward. Checkpoints land in
`output/dex_imitator_inspire_rh_ppo/model.pt`.

## Step 4 — evaluate the trained policy

```bash
python run.py \
    --arg_file args/dex_imitator_inspire_rh_ppo_args.txt \
    --num_envs 16 \
    --mode test \
    --visualize True \
    --model_file output/dex_imitator_inspire_rh_ppo/model.pt \
    --test_episodes 16
```

The viewer shows 16 parallel inspire hands, each replaying the g102 demo
with the trained policy.

## Adding more sequences

1. Drop new pkls under `data/motion_data/grab_demo/<seq>/` (mano_task),
   `…/retargeting/mano2inspire_rh/<seq>_sv_dict.pkl` (retargeted), and
   `…/object/<seq>_obj_traj.pkl` (object trajectory).
2. Edit `data/envs/dex_imitator_inspire_rh_env.yaml`:
   ```yaml
   dataIndices: ["g102", "g117", "g205"]
   ```
3. Re-run training — each parallel env will be randomly assigned one of
   the listed sequences at reset.

## Moving to other embodiments

Every dexhand in `maniptrans_lab/dexhands/` currently still points at the
**ManipTrans-original** URDF path. The only one already updated is
`InspireRH`. To use Allegro / Shadow / XHand / ArtiMANO under Isaac Lab,
you need to:

1. Convert the URDF → USD using Isaac Sim's URDF Importer
   (`data/assets/allegro_hand/...`).
2. Edit the `_urdf_path` in the relevant dexhand class
   (e.g. `dexhands/allegro.py::AllegroRH`).
3. If the MANO→dex retargeting pkls exist at
   `grab_demo/retargeting/mano2allegro_rh/<seq>_sv_dict.pkl`, set
   `dexhand: "allegro"` in the env YAML — the `GrabDemoDexHandRH` loader
   passes `embodiment` through to the pkl path automatically.

## Folder cheat-sheet

```
/home/david/david/Again_0420/
├── data/                                 ← user's pre-provided assets + motion
│   ├── assets/
│   │   ├── inspire_hand/
│   │   │   └── inspire_hand_right_lab_0404/
│   │   │       └── inspire_hand_right_lab.usd       ← used by InspireRH
│   │   ├── mano/rh_mano_lab/rh_mano_lab.usd
│   │   ├── allegro_hand/…
│   │   └── shadow_hand/…
│   └── motion_data/
│       ├── grab_demo/102/102_mano_task.pkl          ← read by GrabDemoDexHandRH
│       ├── grab_demo/retargeting/mano2inspire_rh/
│       │       102_sv_dict.pkl
│       └── grab_demo/object/102_obj_traj.pkl
└── ManipTrans_Lab/                      ← this repo
    ├── maniptrans_lab/
    │   ├── dexhands/inspire.py          ← _urdf_path now points at _lab USD
    │   ├── dataset/grab_demo_dataset.py ← NEW lightweight loader
    │   └── envs/dex_imitator_env.py     ← _pack_data handles dict mano_joints
    ├── data/envs/dex_imitator_inspire_rh_env.yaml   ← dataIndices: ["g102"]
    ├── args/dex_imitator_inspire_rh_ppo_args.txt
    └── tools/smoke_test_dataset.py      ← quick data-only sanity check
```
