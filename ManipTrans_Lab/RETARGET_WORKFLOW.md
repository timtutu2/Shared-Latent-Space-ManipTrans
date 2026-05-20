# Retarget Workflow (Isaac Lab)

This document records the workflow for reproducing Isaac Lab retarget visualization results for GRAB `102`, while keeping them close to the original Gym retarget outputs.

## 1) Key Findings (From This Debug Session)

- As confirmed by `CHECK 1/2` in `view_retarget.py`:
  - DOF name mapping is correct.
  - DOF values passed to simulation are also correct.
- If mesh and skeleton (`opt_joints_pos`) still do not overlap, the root cause is usually **model definition mismatch (URDF/USD)**.
- For Shadow in particular, differences in joint axes/origins between original ManipTrans assets and current Lab assets can prevent exact overlap with simple remapping.

## 2) Recommended Execution Order

### A. First, inspect current retarget output in the viewer

```bash
conda activate maniptrans_lab
cd /home/david/david/Again_0420

python ManipTrans_Lab/view_retarget.py \
  --rh_pkl /home/david/david/Again_0420/data/motion_data/grab_demo/retargeting/mano2shadow_rh/102_sv_dict.pkl \
  --dexhand shadow \
  --shadow_pkl_order auto \
  --device cuda:0
```

### B. For Shadow, re-fit DOFs against the Lab URDF (recommended)

```bash
conda activate maniptrans_lab
cd /home/david/david/Again_0420

python ManipTrans_Lab/tools/retarget_shadow_lab_from_pkl.py \
  --in_pkl /home/david/david/Again_0420/data/motion_data/grab_demo/retargeting/mano2shadow_rh/102_sv_dict.pkl \
  --out_pkl /home/david/david/Again_0420/data/motion_data/grab_demo/retargeting_lab2/mano2shadow_rh/102_sv_dict_fit.pkl \
  --urdf /home/david/david/Again_0420/data/assets/shadow_hand/shadow_hand_woarm_right.urdf \
  --side right \
  --device cuda:0 \
  --iters 300
```

```bash
python ManipTrans_Lab/view_retarget.py \
  --rh_pkl /home/david/david/Again_0420/data/motion_data/grab_demo/retargeting_lab2/mano2shadow_rh/102_sv_dict_fit.pkl \
  --dexhand shadow \
  --shadow_pkl_order urdf_xml \
  --disable_lab_fk_dof \
  --device cuda:0

python ManipTrans_Lab/view_retarget.py \
  --rh_pkl /home/david/david/Again_0420/ManipTrans_Lab/data/motion_data/0f020d_demo/retargeting_lab2/mano2shadow_rh/scene_01__A005++seq__0f020df7ffbbce295312__2023-04-15-14-39-24@0_right_fit.pkl \
  --dexhand shadow \
  --shadow_pkl_order urdf_xml \
  --disable_lab_fk_dof \
  --device cuda:0
```

### C. Lab-URDF refit also supports Inspire/Allegro/ArtiMano

The same script now works for all four right hands:

```bash
python ManipTrans_Lab/tools/retarget_shadow_lab_from_pkl.py \
  --dexhand inspire \
  --in_pkl /home/david/david/Again_0420/data/motion_data/grab_demo/retargeting/mano2inspire_rh/102_sv_dict.pkl \
  --out_pkl /home/david/david/Again_0420/data/motion_data/grab_demo/retargeting_lab2/mano2inspire_rh/102_sv_dict_fit.pkl \
  --side right --device cuda:0 --iters 300
```

```bash
python ManipTrans_Lab/tools/retarget_shadow_lab_from_pkl.py \
  --dexhand allegro \
  --in_pkl /home/david/david/Again_0420/data/motion_data/grab_demo/dexlatent_decoded/allegro/102_sv_dict.pkl \
  --out_pkl /home/david/david/Again_0420/data/motion_data/grab_demo/retargeting_lab2/mano2allegro_rh/102_sv_dict_fit.pkl \
  --side right --device cuda:0 --iters 300
```

```bash
python ManipTrans_Lab/tools/retarget_shadow_lab_from_pkl.py \
  --dexhand artimano \
  --in_pkl /home/david/david/Again_0420/data/motion_data/grab_demo/retargeting/mano2artimano_rh/102_sv_dict.pkl \
  --out_pkl /home/david/david/Again_0420/data/motion_data/grab_demo/retargeting_lab2/mano2artimano_rh/102_sv_dict_fit.pkl \
  --side right --device cuda:0 --iters 300
```

## 3) Per-Hand Retarget Commands (Generate Gym Retarget)

Note: `mano2dexhand.py` below is an Isaac Gym preprocessing script.  
Run it in a `python3.8 + isaacgym` compatible environment.

```bash
conda activate <isaacgym_env>
cd /home/david/david/Again_0420/ManipTrans
```

### Inspire (right)

```bash
python main/dataset/mano2dexhand.py --data_idx g102 --dexhand inspire --side right --headless --iter 2000
```

### Shadow (right)

```bash
python main/dataset/mano2dexhand.py --data_idx g102 --dexhand shadow --side right --headless --iter 3000
```

### ArtiMano (right)

```bash
python main/dataset/mano2dexhand.py --data_idx g102 --dexhand artimano --side right --headless --iter 2000
```

### Allegro (right)

```bash
python main/dataset/mano2dexhand.py --data_idx g102 --dexhand allegro --side right --headless --iter 4000
```

Default output pkl location:
- `data/retargeting/grab_demo/mano2<hand>_rh/102_sv_dict.pkl`

If your viewer/training pipeline reads from `data/motion_data/grab_demo/retargeting/...`, copy the file:

```bash
mkdir -p /home/david/david/Again_0420/data/motion_data/grab_demo/retargeting/mano2shadow_rh
cp /home/david/david/Again_0420/ManipTrans/data/retargeting/grab_demo/mano2shadow_rh/102_sv_dict.pkl \
   /home/david/david/Again_0420/data/motion_data/grab_demo/retargeting/mano2shadow_rh/
```

## 4) Per-Hand Viewer Commands (Isaac Lab)

```bash
conda activate maniptrans_lab
cd /home/david/david/Again_0420
```

### Shadow

```bash
python ManipTrans_Lab/view_retarget.py \
  --rh_pkl /home/david/david/Again_0420/data/motion_data/grab_demo/retargeting_lab2/mano2shadow_rh/102_sv_dict_fit.pkl \
  --dexhand shadow \
  --shadow_pkl_order auto \
  --device cuda:0
```

### Inspire

```bash
python ManipTrans_Lab/view_retarget.py \
  --rh_pkl /home/david/david/Again_0420/data/motion_data/grab_demo/retargeting_lab2/mano2inspire_rh/102_sv_dict_fit.pkl \
  --dexhand inspire \
  --device cuda:0
```

### ArtiMano

```bash
python ManipTrans_Lab/view_retarget.py \
  --rh_pkl /home/david/david/Again_0420/data/motion_data/grab_demo/retargeting/mano2artimano_rh/102_sv_dict.pkl \
  --dexhand artimano \
  --device cuda:0
```

### Allegro

```bash
python ManipTrans_Lab/view_retarget.py \
  --rh_pkl /home/david/david/Again_0420/data/motion_data/grab_demo/retargeting_lab2/mano2allegro_rh/102_sv_dict_fit.pkl \
  --dexhand allegro \
  --device cuda:0
```

## 5) Quick Checklist

- Confirm pkl structure and DOF count from viewer startup logs.
- In `CHECK 2`, verify `pkl→sim` vs `physx` diffs are near zero.
- Inspect `Skeleton vs Articulation` distance distribution.
- If Shadow mismatch is still large, compare with the output of `retarget_shadow_lab_from_pkl.py`.

## 6) Training Workflow (Stage-1 Imitator -> Stage-2 Residual)

Use these from `ManipTrans_Lab/maniptrans_lab`:

```bash
conda activate maniptrans_lab
cd /home/david/david/Again_0420/ManipTrans_Lab/maniptrans_lab
```

### Stage-1 (Shadow RH, GRAB g102)

```bash
python run.py --arg_file ../args/dex_imitator_shadow_rh_ppo_args.txt
```

### Stage-1 (Allegro RH, GRAB g102)

```bash
python run.py --arg_file ../args/dex_imitator_allegro_rh_ppo_args.txt
```

### Stage-2 (Residual, Shadow RH)

1. Edit `data/agents/dex_manip_residual_ppo_agent.yaml`:
   - set `model.residual_base_model_file` to your Stage-1 checkpoint path.

2. Run:

```bash
python run.py --arg_file ../args/dex_manip_sh_shadow_rh_residual_ppo_args.txt
```

### Stage-2 (Residual, Allegro RH)

1. Edit `data/agents/dex_manip_residual_ppo_agent.yaml`:
   - set `model.residual_base_model_file` to your Allegro Stage-1 checkpoint path.

2. Run:

```bash
python run.py --arg_file ../args/dex_manip_sh_allegro_rh_residual_ppo_args.txt
```

### Important Data Path Note (Allegro)

`grab_demo_dataset.py` reads from:
- `data/motion_data/grab_demo/retargeting/mano2allegro_rh/<seq>_sv_dict.pkl`

If your file exists only in `retargeting_lab2`, copy it:

```bash
mkdir -p /home/david/david/Again_0420/data/motion_data/grab_demo/retargeting/mano2allegro_rh
cp /home/david/david/Again_0420/data/motion_data/grab_demo/retargeting_lab2/mano2allegro_rh/102_sv_dict_fit.pkl \
   /home/david/david/Again_0420/data/motion_data/grab_demo/retargeting/mano2allegro_rh/102_sv_dict.pkl
```
