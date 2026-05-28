import time
import numpy as np
import gymnasium as gym
from irl.dataset.expert_dataset import ExpertDataset
from irl.MaxEnt.MaxEnt_simple import MaxEntIRLTrainer_Simple, MaxEntConfig

dataset = ExpertDataset()
dataset.load_trajectories_from_json("data/processed_trajectories_simple.json")

# Register environment
gym.register(
    id='V2GEnv-simple',
    entry_point="irl.envs.V2GEnv_simple:V2GEnv",
    max_episode_steps=96,
)

cfg = MaxEntConfig()
cfg.reward_lr = 0.05
cfg.reward_lr_end = 0.05
cfg.n_epochs = 20
cfg.rollout_samples = 30
cfg.segment = "Male 50-59"
cfg.policy_train_steps_per_iter = 500_000
cfg.folder_name = "MaxEnt/simple/MaxEntIRL_simple_male5059"
cfg.validation = True
cfg.description = "MaxEntIRL_simple_male5059"

trainer = MaxEntIRLTrainer_Simple(
    initial_reward_weights=np.array([0.5, 0.5, -1.0, -1.0, -1.0], dtype=np.float32),
    expert_trajectories=dataset,
    env_name='V2GEnv-simple',
    cfg=cfg,
)

_t0 = time.time()
trainer.train()
_elapsed = time.time() - _t0
print(f"Total training time: {int(_elapsed // 3600)}h {int(_elapsed % 3600 // 60)}m {_elapsed % 60:.1f}s")
