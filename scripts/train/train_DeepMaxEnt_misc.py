"""
Standalone PPO fine-tuner for a frozen Deep MaxEnt discrete reward network.

Loads a reward_net_epoch{N}.pt from a given model folder, trains a fresh PPO
policy under that fixed reward for a configurable number of timesteps, then
saves the policy as  rewardnet_{N}_misc_trained  in the same folder.

Usage (edit the CONFIG block below, then run):
    uv run scripts/train/train_DeepMaxEnt_misc.py
"""

import time
import os
import warnings

import gymnasium as gym
import numpy as np
import torch
from sbx import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from irl.DeepMaxEnt.DeepMaxEnt import RewardNet, PROFIT_OBS_SCALES
from irl.DeepMaxEnt.DeepMaxEnt_discrete import FlattenNormalizeObsWrapper

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------ #
#  CONFIG — edit these fields                                          #
# ------------------------------------------------------------------ #

FOLDER_NAME        = "DeepMaxEnt/discrete/DeepMaxEntIRL_discrete_pricediff_male5059"
REWARD_EPOCH       = 20          # which reward_net_epoch{N}.pt to load
TRAIN_TIMESTEPS    = 3_000_000   # total PPO timesteps to train

# PPO hyperparameters (match the original training config)
POLICY_LR          = 3e-4
POLICY_GAMMA       = 1.0
POLICY_BATCH_SIZE  = 64
POLICY_N_STEPS     = 2048
POLICY_N_EPOCHS    = 10
POLICY_ENT_COEF    = 0.05
N_ENVS             = 4
SEED               = 42

# Reward net architecture (must match what was used during training)
REWARD_OBS_DIM     = 7
REWARD_ACTION_DIM  = 1
REWARD_HIDDEN_DIM  = 32

# Optional scaling applied inside the env (set to match original cfg.reward_scale)
REWARD_SCALE       = 1.0
ACTION_PENALTY     = 0.0

# ------------------------------------------------------------------ #

gym.register(
    id='V2GDeepEnv-discrete',
    entry_point="irl.envs.V2GDeepEnv_discrete:V2GDeepEnv",
    max_episode_steps=96,
)


def main():
    model_dir = f"./models/{FOLDER_NAME}"
    reward_net_path = os.path.join(model_dir, f"reward_net_epoch{REWARD_EPOCH}.pt")
    output_path = os.path.join(model_dir, f"rewardnet_{REWARD_EPOCH}_misc_trained")

    if not os.path.exists(reward_net_path):
        raise FileNotFoundError(f"Reward net not found: {reward_net_path}")

    print(f"Loading reward net from: {reward_net_path}")
    reward_net = RewardNet(
        obs_dim=REWARD_OBS_DIM,
        action_dim=REWARD_ACTION_DIM,
        hidden_dim=REWARD_HIDDEN_DIM,
    )
    reward_net.load_state_dict(torch.load(reward_net_path, weights_only=True))
    reward_net.eval()

    def _make_env():
        e = gym.make('V2GDeepEnv-discrete')
        e = FlattenNormalizeObsWrapper(e)
        return e

    vec_env = DummyVecEnv([_make_env for _ in range(N_ENVS)])
    vec_env = VecMonitor(
        vec_env,
        filename=os.path.join(model_dir, f"monitor_misc_epoch{REWARD_EPOCH}"),
    )

    for e in vec_env.unwrapped.envs:
        e.unwrapped.set_reward_net(reward_net)
        e.unwrapped.set_action_penalty_coeff(ACTION_PENALTY)
        e.unwrapped.set_reward_scale(REWARD_SCALE)
        e.unwrapped.set_initial_states(None)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=POLICY_LR,
        gamma=POLICY_GAMMA,
        batch_size=POLICY_BATCH_SIZE,
        n_steps=POLICY_N_STEPS,
        n_epochs=POLICY_N_EPOCHS,
        ent_coef=POLICY_ENT_COEF,
        seed=SEED,
        tensorboard_log=os.path.join(model_dir, "tensorboard/"),
    )

    print(f"Training PPO for {TRAIN_TIMESTEPS:,} steps under reward_net_epoch{REWARD_EPOCH} ...")
    t0 = time.time()
    model.learn(
        total_timesteps=TRAIN_TIMESTEPS,
        tb_log_name=f"misc_rewardnet_{REWARD_EPOCH}",
        progress_bar=True,
        reset_num_timesteps=True,
        log_interval=10,
    )
    elapsed = time.time() - t0
    print(f"Training time: {int(elapsed // 3600)}h {int(elapsed % 3600 // 60)}m {elapsed % 60:.1f}s")

    model.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
