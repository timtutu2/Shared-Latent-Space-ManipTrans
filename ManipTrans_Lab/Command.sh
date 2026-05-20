cd /home/david/david/Again_0420/ManipTrans_Lab

http://localhost:6006

# Shadow RH - Train (rl_games)
python maniptrans_lab/run_rlgames.py \
  --arg_file args/dex_imitator_shadow_rh_ppo_args.txt \
  --rlg_config data/rl_games/dex_imitator_ppo_rlgames.yaml \
  --mode train \
  --num_envs 4096 \
  --visualize False \
  --save_frequency 500 \
  --out_dir output/dex_imitator_shadow_rh_ppo_rlgames \
  --experiment dex_imitator_shadow_rlgames

python maniptrans_lab/run_rlgames.py \
  --arg_file args/dex_imitator_shadow_rh_ppo_args.txt \
  --rlg_config data/rl_games/dex_imitator_ppo_rlgames.yaml \
  --mode train \
  --num_envs 4096 \
  --visualize False \
  --save_frequency 1000 \
  --out_dir output/dex_imitator_shadow_rh_ppo_rlgames \
  --experiment dex_imitator_shadow_rlgames \
  --model_file output/dex_imitator_shadow_rh_ppo_rlgames/dex_imitator_shadow_rlgames/nn/dex_imitator_shadow_rlgames.pth


tensorboard --logdir output/dex_imitator_shadow_rh_ppo_rlgames --port 6006

# Shadow RH - Test (rl_games)
python maniptrans_lab/run_rlgames.py \
  --arg_file args/dex_imitator_shadow_rh_ppo_args.txt \
  --rlg_config data/rl_games/dex_imitator_ppo_rlgames.yaml \
  --mode test \
  --num_envs 4 \
  --visualize True \
  --model_file output/shadow/dex_imitator_shadow_rh_ppo_rlgames/dex_imitator_shadow_rlgames/nn/dex_imitator_shadow_rlgames.pth

# Allegro RH - Train (rl_games)

tensorboard --logdir output/dex_imitator_allegro_rh_ppo_rlgames --port 6006

# Allegro RH - Test (rl_games)
python maniptrans_lab/run_rlgames.py \
  --arg_file args/dex_imitator_allegro_rh_ppo_args.txt \
  --rlg_config data/rl_games/dex_imitator_ppo_rlgames.yaml \
  --mode test \
  --num_envs 4 \
  --visualize True \
  --model_file output/dex_imitator_allegro_rh_ppo_rlgames/dex_imitator_allegro_rlgames/nn/last_dex_imitator_allegro_rlgames_ep_500_rew_34.016556.pth


# =========================================================
# Stage-2 (Manipulation) with rl_games
# =========================================================
# NOTE:
# - Stage-2 also runs via run_rlgames.py in this setup.
# - It uses dex_manip_sh env configs (shadow/allegro) with rl_games PPO.
# - Residual base model settings from run.py agent config are NOT used by run_rlgames.py.

# table.usd is optional now.
# When table_file is absent, env builds per-env kinematic box tables procedurally.


# Stage-2 Shadow RH - Train (rl_games)
python maniptrans_lab/run_rlgames.py \
  --arg_file args/dex_manip_sh_shadow_rh_residual_ppo_args.txt \
  --rlg_config data/rl_games/dex_manip_residual_ppo_rlgames.yaml \
  --residual_base_model_checkpoint output/dex_imitator_shadow_rh_ppo_rlgames/dex_imitator_shadow_rlgames/nn/dex_imitator_shadow_rlgames.pth \
  --mode train \
  --num_envs 4096 \
  --minibatch_size 1024 \
  --visualize False \
  --save_frequency 500 \
  --out_dir output/dex_manip_sh_shadow_rh_ppo_rlgames \
  --experiment dex_manip_sh_shadow_rlgames

# Stage-2 Shadow RH - Init Visual Check (small envs)
# Use this to visually verify table/object/hand reset and initialization.
python maniptrans_lab/run_rlgames.py \
  --arg_file args/dex_manip_sh_shadow_rh_residual_ppo_args.txt \
  --rlg_config data/rl_games/dex_manip_residual_ppo_rlgames.yaml \
  --residual_base_model_checkpoint output/dex_imitator_shadow_rh_ppo_rlgames/dex_imitator_shadow_rlgames/nn/dex_imitator_shadow_rlgames.pth \
  --mode train \
  --num_envs 4096 \
  --visualize False \
  --save_frequency 500 \
  --out_dir output/dex_manip_sh_shadow_rh_ppo_rlgames \
  --experiment dex_manip_sh_shadow_rlgames

tensorboard --logdir output/dex_manip_sh_shadow_rh_ppo_rlgames --port 6007

# Stage-2 Shadow RH - Test (rl_games)
python maniptrans_lab/run_rlgames.py \
  --arg_file args/dex_manip_sh_shadow_rh_residual_ppo_args.txt \
  --rlg_config data/rl_games/dex_manip_residual_ppo_rlgames.yaml \
  --residual_base_model_checkpoint output/shadow/dex_imitator_shadow_rh_ppo_rlgames/dex_imitator_shadow_rlgames/nn/dex_imitator_shadow_rlgames.pth \
  --mode test \
  --num_envs 4 \
  --visualize True \
  --model_file output/shadow/dex_manip_sh_shadow_rh_ppo_rlgames/dex_manip_sh_shadow_rlgames/nn/dex_manip_sh_shadow_rlgames.pth


# Stage-2 Allegro RH - Train (rl_games)
python maniptrans_lab/run_rlgames.py \
  --arg_file args/dex_manip_sh_allegro_rh_residual_ppo_args.txt \
  --rlg_config data/rl_games/dex_manip_residual_ppo_rlgames.yaml \
  --mode train \
  --num_envs 4096 \
  --minibatch_size 1024 \
  --visualize False \
  --save_frequency 500 \
  --out_dir output/dex_manip_sh_allegro_rh_ppo_rlgames \
  --experiment dex_manip_sh_allegro_rlgames

# Stage-2 Allegro RH - Init Visual Check (small envs)
python maniptrans_lab/run_rlgames.py \
  --arg_file args/dex_manip_sh_allegro_rh_residual_ppo_args.txt \
  --rlg_config data/rl_games/dex_manip_residual_ppo_rlgames.yaml \
  --mode test \
  --num_envs 4 \
  --visualize True \
  --games_num 10 \
  --print_stats False

# Stage-2 Allegro RH - Test (rl_games)
python maniptrans_lab/run_rlgames.py \
  --arg_file args/dex_manip_sh_allegro_rh_residual_ppo_args.txt \
  --rlg_config data/rl_games/dex_manip_residual_ppo_rlgames.yaml \
  --mode test \
  --num_envs 4 \
  --visualize True \
  --model_file output/dex_manip_sh_allegro_rh_ppo_rlgames/dex_manip_sh_allegro_rlgames/nn/dex_manip_sh_allegro_rlgames.pth
