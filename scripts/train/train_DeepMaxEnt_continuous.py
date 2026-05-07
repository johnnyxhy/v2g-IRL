import time
import gymnasium as gym
import numpy as np
import torch
from irl.DeepMaxEnt.DeepMaxEnt_continuous import DeepMaxEntConfig, DeepMaxEntIRLTrainer, load_deep_expert_data
import warnings

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
    # Load expert data (run expert_loader_deep_continuous.py first to generate the JSON)
    train_set, val_set = load_deep_expert_data(
        "data/processed_trajectories_deep_profit.json",
        segment="Male 50-59",
        train_ratio=0.8,
    )

    cfg = DeepMaxEntConfig()
    cfg.n_epochs = 30
    cfg.reward_lr = 1e-3
    cfg.reward_lr_end = 1e-3
    cfg.rollout_samples = 30
    cfg.policy_train_steps_per_iter = 100_000
    cfg.reward_hidden_dim = 32
    cfg.segment = "Male 50-59"
    cfg.folder_name = "DeepMaxEnt/continuous/DeepMaxEntIRL_continuous_male5059"
    cfg.validation = True
    cfg.action_penalty_coeff = 0.0

    # Warm-start reward network from a previous run (set to None to train from scratch)
    #cfg.pretrained_reward_net_path = "models/DeepMaxEnt/continuous/DeepMaxEntIRL_continuous_male5059/reward_net_epoch30.pt"

    trainer = DeepMaxEntIRLTrainer(
        train_set=train_set,
        val_set=val_set,
        env_name='V2GDeepEnv-continuous',
        cfg=cfg,
    )

    _t0 = time.time()
    trainer.train()
    _elapsed = time.time() - _t0
    print(f"Total training time: {int(_elapsed // 3600)}h {int(_elapsed % 3600 // 60)}m {_elapsed % 60:.1f}s")
