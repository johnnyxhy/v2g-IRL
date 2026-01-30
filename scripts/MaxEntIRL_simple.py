import numpy as np
import gymnasium as gym
from irl.dataset.expert_dataset_simple import ExpertDatasetSimple
from irl.MaxEnt.MaxEnt_simple import MaxEntIRLTrainer_Simple, MaxEntConfig

dataset = ExpertDatasetSimple()
dataset.load_trajectories_from_json("data/processed_trajectories_simple.json")

# Register environment
gym.register(
    id='V2GEnv-simple',
    entry_point="irl.envs.V2GEnv_simple:V2GEnv",
    max_episode_steps=96,
)

cfg = MaxEntConfig()
cfg.reward_lr = 0.5
cfg.n_epochs = 20
cfg.rollout_samples = 10
cfg.segment = "Male 50-59"
cfg.policy_train_steps_per_iter = 10_000
cfg.folder_name = "MaxEntIRL_simple_exp4_2"
cfg.validation = True

trainer = MaxEntIRLTrainer_Simple(
    initial_reward_weights= np.array([-1, 0, -1, -1], dtype=np.float32),
    expert_trajectories=dataset,
    env_name='V2GEnv-simple',
    cfg=cfg,
)

trainer.train()

