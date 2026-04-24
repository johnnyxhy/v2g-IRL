import time
import numpy as np
import gymnasium as gym
from irl.dataset.expert_dataset import ExpertDataset
from irl.MaxEnt.MaxEnt_continuous import MaxEntIRLTrainer_Continuous, MaxEntConfig
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

dataset = ExpertDataset()
dataset.load_trajectories_from_json("data/processed_trajectories_profit.json")
# Register environment
gym.register(
    id='V2GEnv-profit',
    entry_point="irl.envs.V2GEnv_profit:V2GEnv",
    max_episode_steps=96,
)

if __name__ == "__main__":
    cfg = MaxEntConfig()
    cfg.reward_lr = 1e-1
    cfg.reward_lr_end = 1e-1
    cfg.n_epochs = 50
    cfg.rollout_samples = 20
    cfg.segment = "Male 50-59"
    cfg.policy_train_steps_per_iter = 200_000
    cfg.folder_name = "MaxEntIRL_profit_v7_exp2"
    cfg.validation = True

    trainer = MaxEntIRLTrainer_Continuous(
        initial_reward_weights= np.array([0.8, 0.8, -0.5, -0.5, -1.0, -0.2, -1.0], dtype=np.float32),
        expert_trajectories=dataset,
        env_name='V2GEnv-profit',
        cfg=cfg,
    )

    _t0 = time.time()
    trainer.train()
    _elapsed = time.time() - _t0
    print(f"Total training time: {int(_elapsed // 3600)}h {int(_elapsed % 3600 // 60)}m {_elapsed % 60:.1f}s")

