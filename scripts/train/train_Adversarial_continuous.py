import time
import gymnasium as gym
import numpy as np
import torch
import warnings

from irl.Adversarial.Adversarial_continuous import (
    AIRLConfig,
    AIRLTrainer,
    load_airl_expert_data,
)

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

gym.register(
    id='V2GDeepEnv-continuous',
    entry_point="irl.envs.V2GDeepEnv_continuous:V2GDeepEnv",
    max_episode_steps=96,
)

if __name__ == "__main__":
    # Run expert_loader_airl_continuous.py first to generate the JSON:
    #   python -m irl.dataset.expert_loader_airl_continuous
    train_set, val_set = load_airl_expert_data(
        "data/processed_trajectories_airl_continuous.json",
        segment="Male 50-59",
        train_ratio=0.8,
    )

    cfg = AIRLConfig()

    # ---- IRL outer loop ----
    cfg.n_epochs = 50
    cfg.disc_lr = 1e-3
    cfg.disc_lr_end = 1e-3
    cfg.rollout_samples = 30
    cfg.disc_epochs = 1
    cfg.disc_l2_reg = 0.01

    # ---- Network dimensions ----
    cfg.reward_hidden_dim = 32
    cfg.shaping_hidden_dim = 32

    # ---- SAC inner loop ----
    cfg.policy_train_steps_per_iter = 100_000
    cfg.policy_train_lr = 3e-4
    cfg.policy_gamma = 1.0          # consistent with γ^Δt shaping term
    cfg.policy_batch_size = 64
    cfg.policy_buffer_size = 100_000
    cfg.policy_learning_starts = 1_000
    cfg.policy_ent_coef = 'auto'

    # ---- Reward scaling and penalties ----
    cfg.reward_scale = 0.2          # keep SAC reward signal in a reasonable range
    cfg.action_penalty_coeff = 0.0

    # ---- BC pre-training ----
    cfg.bc_pretrain_steps = 5_000
    cfg.bc_lr = 1e-3

    # ---- Data ----
    cfg.segment = "Male 50-59"
    cfg.train_ratio = 0.8

    # ---- Misc ----
    cfg.n_envs = 1
    cfg.validation = True
    cfg.folder_name = "Adversarial/continuous/AIRL_continuous_male5059"
    cfg.description = (
        "AIRL continuous (profit env), SAC inner loop.\n"
        "Segment: Male 50-59, 50 epochs, 100k SAC steps/epoch.\n"
        "reward_scale=0.2, disc_epochs=1, bc_pretrain_steps=5000.\n"
        "gamma=1.0, ent_coef=auto.\n"
    )

    trainer = AIRLTrainer(
        train_set=train_set,
        val_set=val_set,
        env_name='V2GDeepEnv-continuous',
        cfg=cfg,
    )

    _t0 = time.time()
    trainer.train()
    _elapsed = time.time() - _t0
    print(f"Total training time: {int(_elapsed // 3600)}h {int(_elapsed % 3600 // 60)}m {_elapsed % 60:.1f}s")
