import time
import gymnasium as gym
import numpy as np
import torch
from irl.DeepMaxEnt.DeepMaxEnt_discrete import (
    DeepMaxEntDiscreteConfig,
    DeepMaxEntDiscreteTrainer,
    load_deep_discrete_expert_data,
)
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

gym.register(
    id='V2GDeepEnv-discrete',
    entry_point="irl.envs.V2GDeepEnv_discrete:V2GDeepEnv",
    max_episode_steps=96,
)

if __name__ == "__main__":
    train_set, val_set = load_deep_discrete_expert_data(
        "data/processed_trajectories_deep_discrete_pricediff.json",
        segment="Male 40-49",
        train_ratio=0.8,
    )

    cfg = DeepMaxEntDiscreteConfig()
    cfg.n_epochs = 20
    cfg.reward_lr = 1e-3
    cfg.reward_lr_end = 1e-3
    cfg.rollout_samples = 30
    cfg.policy_train_steps_per_iter = 1_500_000
    cfg.policy_n_steps = 2048
    cfg.policy_n_epochs = 10
    cfg.policy_ent_coef = 0.01
    cfg.reset_ppo_each_epoch = True
    cfg.reward_hidden_dim = 32
    cfg.reward_grad_clip = 5.0
    cfg.reward_l2_reg = 0.01
    cfg.folder_name = "DeepMaxEnt/discrete/DeepMaxEntIRL_discrete_pricediff_male4049"
    cfg.validation = True
    cfg.action_penalty_coeff = 0.0
    cfg.reward_scale = 10.0

    cfg.description = "Added 0.2 buffer, fixed inconsistency in soc_target. Running with new dataset, 20 epochs, 1.5M policy steps per epoch, 0.01 entropy coeff, 0.01 reward L2 reg, 5.0 grad clip, reward scale 10.0, no action penalty."

    # Warm-start reward network (set to None to train from scratch)
    #cfg.pretrained_reward_net_path = "./models/DeepMaxEnt/discrete/DeepMaxEntIRL_discrete_pricediff_male5059_v2/reward_net_epoch16.pt"

    trainer = DeepMaxEntDiscreteTrainer(
        train_set=train_set,
        val_set=val_set,
        env_name='V2GDeepEnv-discrete',
        cfg=cfg,
    )
    _t0 = time.time()
    trainer.train()
    _elapsed = time.time() - _t0
    print(f"Total training time: {int(_elapsed // 3600)}h {int(_elapsed % 3600 // 60)}m {_elapsed % 60:.1f}s")
