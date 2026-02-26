import numpy as np
import gymnasium as gym
from irl.dataset.expert_dataset import ExpertDataset
from irl.MaxEnt.MaxEnt_continuous import MaxEntIRLTrainer_Continuous, MaxEntConfig
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

dataset = ExpertDataset()
dataset.load_trajectories_from_json("data/processed_trajectories_continuous.json")
# Register environment
gym.register(
    id='V2GEnv-continuous',
    entry_point="irl.envs.V2GEnv_continuous:V2GEnv",
    max_episode_steps=96,
)

if __name__ == "__main__":
    cfg = MaxEntConfig()
    cfg.reward_lr = 1
    cfg.reward_lr_end = 0.01 # Decay reward LR from 1 to 0.01 over training — allows larger initial
    cfg.n_epochs = 5
    cfg.rollout_samples = 20
    cfg.segment = "Female 50-59"
    cfg.policy_train_steps_per_iter = 100_000
    cfg.folder_name = "MaxEntIRL_continuous_v7_exp6(Female 50-59)"
    cfg.validation = True

    trainer = MaxEntIRLTrainer_Continuous(
        initial_reward_weights= np.array([0.7, 0.7, -1, -1, -1], dtype=np.float32),
        expert_trajectories=dataset,
        env_name='V2GEnv-continuous',
        cfg=cfg,
    )

    trainer.train()

