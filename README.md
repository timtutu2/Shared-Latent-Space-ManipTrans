# DexLatent

## Installation
```bash
conda env create -f environment.yml
conda activate maniptrans_lab
```

## Project Structure
- `Checkpoints/`: Saved model checkpoints
- `decoded_pkls/`: Decoded PKL outputs
- `dexlatent/`: Source code
- `eval/`: Evaluation and visualization scripts
- `environment.yml`: Conda environment definition

## Start Training
Run from the repository root:

```bash
python -m dexlatent.train
```

Default checkpoint output directory is inside:
`Checkpoints/dexlatent/<timestamp>/`

Example (set custom hands / steps):
```bash
python -m dexlatent.train --hands mano_right shadow_right inspire_right --num_steps 5000
```

## Add a New Hand
Use an existing checkpoint and train only the new hand:
```bash
python -m dexlatent.add_hand_train \
  --ckpt Checkpoints/dexlatent/dexlatent_arti_sha_ins/checkpoint_epoch_5000.pt \
  --new_hand allegro_right \
  --num_steps 5000
```

## Check Results / Visualize
1) Orienting mouse
```bash
python -m dexlatent.visualize_isaaclab \
  --ckpt Checkpoints/dexlatent/20260412_231724/checkpoint_epoch_10000.pt \
  --hands mano_right shadow_right inspire_right allegro_right \
  --source mano_right \
  --motion_pkl ../data/motion_data/grab_demo/retargeting/mano2artimano_rh/102_sv_dict.pkl \
  --spacing 0.3
```

2) Stir with spoon
```bash
python -m dexlatent.visualize_isaaclab \
    --ckpt Checkpoints/dexlatent/allegro_right_20260412_162420/checkpoint_epoch_10000.pt \
    --hands mano_right shadow_right inspire_right allegro_right \
    --source mano_right \
    --motion_pkl ../data/motion_data/0f020d_demo/mano2artimano/scene_01__A005++seq__0f020df7ffbbce295312__2023-04-15-14-39-24@0_right.pkl \
    --object_pkl ../data/motion_data/0f020d_demo/object/O02_0030_00002_obj_traj.pkl \
    --object_usd ../data/motion_data/0f020d_demo/object/scan.usd \
    --object_scale 1.0 \
    --spacing 0.3
```

3) Cutting bread with knife
```bash
python -m dexlatent.visualize_isaaclab \
    --ckpt Checkpoints/dexlatent/allegro_right_20260412_162420/checkpoint_epoch_10000.pt \
    --hands mano_left mano_right shadow_left shadow_right \
    --source mano_left \
    --motion_pkl ../data/motion_data/083f7a_demo/artimano/scene_01__A001++seq__083f7a577484ba7929a9__2023-04-27-19-25-24@0_left.pkl \
    --source_right mano_right \
    --motion_pkl_right ../data/motion_data/083f7a_demo/artimano/scene_01__A001++seq__083f7a577484ba7929a9__2023-04-27-19-25-24@0_right.pkl \
    --object_usd ../data/motion_data/083f7a_demo/object/S20005/model_align.usd \
    --object_usd_right ../data/motion_data/083f7a_demo/object/O02@0094@00004/scan.usd \
    --scene_pkl ../data/motion_data/083f7a_demo/scene_01__A001++seq__083f7a577484ba7929a9__2023-04-27-19-25-24.pkl \
    --program_json ../data/motion_data/083f7a_demo/gym_translate/scene_01__A001++seq__083f7a577484ba7929a9__2023-04-27-19-25-24.json \
    --stage 0 \
    --subsample 4 \
    --playback_speed 1.0 \
    --spacing 0.5 \
    --shadow_shift_y 0.6
```
