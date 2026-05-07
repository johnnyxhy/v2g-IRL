import time
import gymnasium as gym
import numpy as np
import torch
from irl.Adversarial.Adversarial_discrete import (
    AIRLConfig,
    AIRLTrainer,
    load_airl_expert_data,
)
import warnings

#warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

gym.register(
    id='V2GDeepEnv-discrete',
    entry_point="irl.envs.V2GDeepEnv_discrete:V2GDeepEnv",
    max_episode_steps=96,
)

if __name__ == "__main__":
    train_set, val_set = load_airl_expert_data(
        "data/processed_trajectories_airl_discrete_pricediff.json",
        segment="Male 40-49",
        train_ratio=0.8,
    )

    cfg = AIRLConfig()
    cfg.n_epochs = 20
    cfg.disc_lr = 1e-3
    cfg.disc_lr_end = 1e-3
    cfg.rollout_samples = 30
    cfg.policy_train_steps_per_iter = 1_500_000
    # γ=1.0: SBX PPO applies its discount once per *action* regardless of Δt.
    # Using γ<1 would mismatch the γ^Δt shaping term in _compute_f, breaking the
    # AIRL disentanglement guarantee.  With γ=1.0, γ^Δt=1 everywhere and PPO and
    # the discriminator both use undiscounted returns — consistent and correct for
    # this finite-horizon episodic task.
    cfg.policy_gamma = 1.0
    cfg.policy_n_steps = 2048
    cfg.policy_n_epochs = 10
    cfg.policy_ent_coef = 0.01  
    cfg.reset_ppo_each_epoch = True
    cfg.reward_hidden_dim = 32
    cfg.shaping_hidden_dim = 32
    cfg.disc_l2_reg = 0.01
    cfg.disc_epochs = 1           # 75 grad steps/IRL epoch — slows disc convergence so PPO can adapt
    cfg.action_penalty_coeff = 0.0
    cfg.reward_scale = 0.2
    cfg.bc_pretrain_steps = 5000  # BC pre-training: ~3min, bootstraps policy near expert before AIRL
    cfg.bc_lr = 1e-3
    cfg.validation = True
    cfg.n_envs = 4
    cfg.folder_name = "Adversarial/discrete/Adversarial_discrete_male4049"

    cfg.description = (
        "Segment: Male 40-49. 20 epochs, 1.5M PPO steps/epoch, "
        "BC pre-training 5000 steps (key fix: policy started from random init, "
        "never visited expert states, discriminator couldn't provide useful gradient). "
        "reward_scale=0.2, disc_epochs=1, ent_coef=0.1, reset_ppo=True, gamma=1.0."
    )

    # Warm-start from a previous run (set to None to train from scratch)
    cfg.pretrained_reward_net_path = None
    cfg.pretrained_shaping_net_path = None

    trainer = AIRLTrainer(
        train_set=train_set,
        val_set=val_set,
        env_name='V2GDeepEnv-discrete',
        cfg=cfg,
    )
    _t0 = time.time()
    trainer.train()
    _elapsed = time.time() - _t0
    print(f"Total training time: {int(_elapsed // 3600)}h {int(_elapsed % 3600 // 60)}m {_elapsed % 60:.1f}s")
