# ManipTrans_Lab

Isaac-Lab port of [ManipTrans](https://github.com/ManipTrans/ManipTrans), laid out
in the style of [MimicKit](https://github.com/MimicKit/MimicKit).

* **Content** (dexhands, datasets, task observations/rewards, reset schedules)
  comes from ManipTrans.
* **Structure** (engine / env / learning split, YAML-driven builders,
  `args/*.txt` entrypoints) comes from MimicKit.
* The simulator backend is Isaac Lab (via MimicKit's `IsaacLabEngine`); the
  original Isaac Gym implementation lives in the sibling `ManipTrans/` repo.

## Layout

```
ManipTrans_Lab/
├── maniptrans_lab/
│   ├── run.py                    # entrypoint  (mirrors mimickit/run.py)
│   ├── engines/                  # physics-sim abstraction
│   │   ├── engine.py             # MimicKit base class (ObjType, ControlMode)
│   │   ├── isaac_lab_engine.py   # MimicKit's Isaac Lab engine (unchanged)
│   │   ├── isaac_lab_recorder.py
│   │   └── engine_builder.py     # dispatches on engine_name
│   ├── envs/                     # task environments
│   │   ├── base_env.py           # MimicKit BaseEnv (unchanged)
│   │   ├── sim_env.py            # MimicKit SimEnv (unchanged)
│   │   ├── dex_env.py            # NEW: base dexhand-on-table env
│   │   ├── dex_imitator_env.py   # PORT: ManipTrans dexhandimitator
│   │   ├── dex_manip_sh_env.py   # PORT: ManipTrans dexhandmanip_sh
│   │   ├── dex_manip_bih_env.py  # PORT: ManipTrans dexhandmanip_bih
│   │   └── env_builder.py        # dispatches on env_name
│   ├── dexhands/                 # ManipTrans dexhand wrappers (copied)
│   │   ├── base.py, factory.py, decorators.py
│   │   └── inspire.py, allegro.py, shadow.py, xhand.py,
│   │       artimano.py, inspireftp.py
│   ├── dataset/                  # ManipTrans demo-data loaders (copied)
│   │   ├── base.py, factory.py, decorators.py, transform.py
│   │   ├── grab_dataset_dexhand.py
│   │   ├── oakink2_dataset_dexhand_{rh,lh}.py
│   │   ├── oakink2_dataset_utils.py
│   │   ├── oakink2_layer/
│   │   └── mano2dexhand.py       # isaacgym lazily imported (preprocessing)
│   ├── learning/                 # RL stack
│   │   ├── base_agent.py, base_model.py           (MimicKit, unchanged)
│   │   ├── ppo_agent.py, ppo_model.py             (MimicKit, unchanged)
│   │   ├── dict_ppo_agent.py                      (NEW: dict-obs PPO)
│   │   ├── dict_ppo_model.py                      (NEW)
│   │   ├── dict_normalizer.py                     (NEW)
│   │   ├── experience_buffer.py, normalizer.py,
│   │   │   distribution_*.py, return_tracker.py,
│   │   │   mp_optimizer.py, rl_util.py, dummy_agent.py (MimicKit, unchanged)
│   │   ├── agent_builder.py      # dispatches on agent_name (adds "DictPPO")
│   │   └── nets/
│   │       ├── net_builder.py
│   │       ├── fc_2layers_{256,512,1024}units.py  (MimicKit)
│   │       └── dict_feature_fusion.py             (NEW: ports SimpleFeatureFusion)
│   ├── util/                     # MimicKit utilities (copied verbatim)
│   └── utils/                    # ManipTrans utilities (torch_jit, pose, transforms)
│
├── data/
│   ├── engines/isaac_lab_engine.yaml
│   ├── envs/dex_imitator_inspire_rh_env.yaml
│   ├── envs/dex_manip_sh_inspire_rh_env.yaml
│   ├── envs/dex_manip_bih_inspire_env.yaml
│   └── agents/dex_imitator_ppo_agent.yaml
│       agents/dex_manip_ppo_agent.yaml
│
├── args/
│   ├── dex_imitator_inspire_rh_ppo_args.txt
│   ├── dex_manip_sh_inspire_rh_ppo_args.txt
│   └── dex_manip_bih_inspire_ppo_args.txt
│
└── assets/                       # put ManipTrans URDFs + USDs here
```

## Installation

```bash
# 1. Isaac Lab (see https://isaac-sim.github.io/IsaacLab/)
# 2. Python deps
pip install -r requirements.txt
```

## Running

Launch training the MimicKit way — feed an args file to `run.py`:

```bash
cd maniptrans_lab
python run.py --arg_file ../args/dex_imitator_inspire_rh_ppo_args.txt
```

Override on the command line:

```bash
python run.py --arg_file ../args/dex_imitator_inspire_rh_ppo_args.txt \
              --num_envs 1024 --mode test --visualize
```

## Architecture notes

### Engine abstraction

All physics calls go through `engines.engine.Engine` — ManipTrans_Lab's tasks
**never** import `isaaclab` directly. The engine exposes:

* `create_env()`, `create_obj(..., obj_type=ObjType.articulated|rigid)`
* `step()`, `render()`, `initialize_sim()`
* state getters: `get_{root,dof,body}_{pos,rot,vel,ang_vel}` / `get_contact_forces`
* state setters: `set_{root,dof,body}_{pos,rot,vel,ang_vel}`
* `set_cmd(obj_id, dof_cmd)` – position / velocity / torque targets
* `find_obj_body_id(obj_id, name)` – look up a body by URDF link name

### Dict observations

ManipTrans's policy sees three streams:

* **proprioception** — joint angles (q, cosq, sinq) + wrist base_state (pos masked)
* **privileged** — joint velocities, and for manipulation tasks also
  object pose/vel/com/weight + fingertip contact forces
* **target** — future demo wrist pose/vel + future joint positions

`DictPPOAgent` keeps the MimicKit PPO update rules but swaps in a
`DictNormalizer` (one `Normalizer` per key) and `DictPPOModel` (two dict →
MLP towers for actor and critic, with optional asymmetric obs selection —
actor = `proprioception+target`, critic = `proprioception+privileged+target`).

The actor network is `dict_feature_fusion` — a MimicKit-style net module
that mirrors ManipTrans's `SimpleFeatureFusion` (Identity extractors per
stream → concatenate → MLP head).

### What got ported and what was left

| ManipTrans file                                       | ManipTrans_Lab file |
| ----------------------------------------------------- | ---------------------- |
| `maniptrans_envs/lib/envs/core/vec_task.py`           | subsumed by `envs/sim_env.py` + `envs/dex_env.py` |
| `maniptrans_envs/lib/envs/core/sim_config.py`         | engine YAML (`data/engines/isaac_lab_engine.yaml`) |
| `maniptrans_envs/lib/envs/dexhands/*`                 | `maniptrans_lab/dexhands/*` (copied) |
| `maniptrans_envs/lib/envs/tasks/dexhandimitator.py`   | `envs/dex_imitator_env.py` |
| `maniptrans_envs/lib/envs/tasks/dexhandmanip_sh.py`   | `envs/dex_manip_sh_env.py` |
| `maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py`  | `envs/dex_manip_bih_env.py` |
| `main/dataset/*`                                      | `maniptrans_lab/dataset/*` (copied, imports rewired) |
| `main/dataset/mano2dexhand.py`                        | kept; `isaacgym` imported lazily (preprocessing) |
| `main/rl/train.py` (hydra + rl_games)                 | `run.py` (MimicKit PPO) + `run_rlgames.py` (rl_games compatibility) |
| `lib/rl/* (rl_games network builders)`                | `learning/dict_ppo_agent.py` + `dict_ppo_model.py` + `rlgames/*` |
| `lib/nn/features/fusion.py` (SimpleFeatureFusion)     | ported to `learning/nets/dict_feature_fusion.py` |
| `lib/utils/torch_jit_utils.py`, `pose_utils.py`, …    | `maniptrans_lab/utils/` (copied) |

Sections in task files marked `# PORT:` are 1-to-1 ports of Isaac Gym code;
sections marked `# TODO(port):` expose a hook where the original ManipTrans
`@torch.jit.script` reward function (e.g. `compute_imitation_reward`) should
be pasted in verbatim once the call sites are wired up.

### rl_games compatibility mode

To stay close to original ManipTrans training stack, this repo also provides
`maniptrans_lab/run_rlgames.py` plus `maniptrans_lab/rlgames/*`:

- `dict_obs_actor_critic` network builder with ManipTrans-style dict feature fusion
- `my_continuous_a2c_logstd` model wrapper (running-mean/std for dict obs)
- `data/rl_games/dex_imitator_ppo_rlgames.yaml` with ManipTrans-like PPO hyperparameters

Example:

```bash
python maniptrans_lab/run_rlgames.py \
  --arg_file args/dex_imitator_allegro_rh_ppo_args.txt \
  --rlg_config data/rl_games/dex_imitator_ppo_rlgames.yaml \
  --mode train --num_envs 4096 --visualize False
```

### mano2dexhand (MANO → dexhand retargeting)

Retargeting is a **preprocessing** step: run the original
`ManipTrans/main/dataset/mano2dexhand.py` against your GRAB / OakInk2 data
to produce `.pkl` files, then point ManipTrans_Lab's dataset loaders at
those files via `dataIndices`. ManipTrans_Lab doesn't need Isaac Gym at
runtime.

### Asset files

Put your USD-converted dexhand URDFs and scene assets under `assets/`.
Isaac Lab's engine calls `UsdFileCfg(usd_path=...)` internally — so `.urdf`
paths in `DexHand._urdf_path` are auto-converted to `.usd` by
`IsaacLabEngine._parse_usd_path()`; make sure a USD exists next to each URDF.
