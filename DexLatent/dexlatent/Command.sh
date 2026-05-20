python -m dexlatent.train

# python -m dexlatent.visualize_isaaclab \
#     --ckpt Checkpoints/dexlatent/20260412_231724/checkpoint_epoch_10000.pt \
#     --hands mano_right shadow_right inspire_right\
#     --source mano_right \
#     --motion_pkl data/motion_data/grab_demo/retargeting/mano2artimano_rh/102_sv_dict.pkl \
#     --spacing 0.3

# python -m dexlatent.visualize_isaaclab \
#     --ckpt Checkpoints/dexlatent/20260412_154813/checkpoint_epoch_10000.pt \
#     --hands mano_right shadow_right\
#     --source mano_right \
#     --motion_pkl data/motion_data/grab_demo/retargeting/mano2artimano_rh/102_sv_dict.pkl \
#     --spacing 0.3


python -m dexlatent.visualize_isaaclab \
  --ckpt Checkpoints/dexlatent/20260412_231724/checkpoint_epoch_10000.pt \
  --hands mano_right shadow_right inspire_right \
  --source mano_right \
  --motion_pkl ../data/motion_data/grab_demo/retargeting/mano2artimano_rh/102_sv_dict.pkl \
  --spacing 0.3


python -m dexlatent.pre_decode_retarget \
  --ckpt Checkpoints/dexlatent/allegro_right_20260412_162420/checkpoint_epoch_10000.pt \
  --source_hand mano_right \
  --target_hand shadow_right \
  --src_pkl ../data/motion_data/grab_demo/retargeting/mano2artimano_rh/102_sv_dict.pkl \
  --out ../data/motion_data/grab_demo/dexlatent_decoded/shadow/102_sv_dict.pkl \
  --device cuda:0

## DexLatent RL Training (Shadow hand, latent-space policy)
python mainptrans/run.py \
    --mode train --num_envs 4096 \
    --engine_config data/engines/isaac_lab_engine.yaml \
    --env_config data/envs/shadow_dexlatent/shadow_dexlatent_env.yaml \
    --agent_config data/agents/shadow_dexlatent/dexhand_imitator_agent.yaml \
    --visualize false \
    --out_dir output/shadow_dexlatent/imitator_102 \
    --devices cuda:0 --max_samples 500000000

## Adding Allegro

python -m dexlatent.visualize_isaaclab \
    --ckpt Checkpoints/dexlatent/allegro_right_20260412_162420/checkpoint_epoch_10000.pt \
    --hands mano_right shadow_right inspire_right allegro_right \
    --source mano_right \
    --motion_pkl data/motion_data/grab_demo/retargeting/mano2artimano_rh/102_sv_dict.pkl \
    --spacing 0.3


python -m dexlatent.visualize_isaaclab \
    --ckpt Checkpoints/dexlatent/allegro_right_20260412_162420/checkpoint_epoch_10000.pt \
    --hands mano_right shadow_right inspire_right allegro_right \
    --source inspire_right \
    --motion_pkl data/motion_data/grab_demo/dexlatent_decoded/inspire/102_sv_dict.pkl \
    --spacing 0.3
