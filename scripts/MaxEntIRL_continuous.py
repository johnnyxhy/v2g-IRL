import numpy as np
import gymnasium as gym
from irl.dataset.expert_dataset_continuous import ExpertDatasetContinuous
from irl.MaxEnt.MaxEnt_continuous import MaxEntIRLTrainer_Continuous, MaxEntConfig

dataset = ExpertDatasetContinuous()
dataset.load_trajectories_from_json("data/processed_trajectories_continuous.json")
# Register environment
gym.register(
    id='V2GEnv-continuous',
    entry_point="irl.envs.V2GEnv_continuous:V2GEnv",
    max_episode_steps=96,
)

cfg = MaxEntConfig()
cfg.reward_lr = 0.5
cfg.n_epochs = 20
cfg.rollout_samples = 5
cfg.segment = "Male 50-59"
cfg.policy_train_steps_per_iter = 20_000
cfg.folder_name = "MaxEntIRL_continuous_exp1"
cfg.validation = True

trainer = MaxEntIRLTrainer_Continuous(
    initial_reward_weights= np.array([-1, 0, -1, -1], dtype=np.float32),
    expert_trajectories=dataset,
    env_name='V2GEnv-continuous',
    cfg=cfg,
)

trainer.train()

