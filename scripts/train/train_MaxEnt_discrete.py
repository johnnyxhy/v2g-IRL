import time
import numpy as np
import gymnasium as gym
from irl.dataset.expert_dataset import ExpertDataset
from irl.MaxEnt.MaxEnt_discrete import MaxEntIRLTrainer_Discrete_PPO, MaxEntConfig
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

dataset = ExpertDataset()
dataset.load_trajectories_from_json("data/processed_trajectories_discrete_pricediff.json")

gym.register(
    id='V2GEnv-discrete',
    entry_point="irl.envs.V2GEnv_discrete:V2GEnv",
    max_episode_steps=96,
)

if __name__ == "__main__":
    cfg = MaxEntConfig()
    cfg.reward_lr = 0.01
    cfg.reward_lr_end = 0.01
    cfg.n_epochs = 20
    cfg.rollout_samples = 20
    cfg.grad_clip_norm = 1.0
    cfg.segment = "Male 50-59"
    cfg.policy_train_steps_per_iter = 1_500_000
    cfg.policy_n_steps = 2048
    cfg.policy_n_epochs = 10
    cfg.policy_ent_coef = 0.01
    cfg.folder_name = "MaxEnt/discrete/MaxEntIRL_discrete_pricediff_male5059"
    cfg.validation = True
    cfg.reset_ppo_each_epoch = True
    cfg.reg_lambda = 0.01
    cfg.description = "new soc_target, 20 epochs, 1,500,000 PPO steps, PPO reset, 0.01 boundary penalty, 0.01 l2 reg, 0.01 lr"

    trainer = MaxEntIRLTrainer_Discrete_PPO(
        initial_reward_weights=np.array([1.0, 0.9, -0.1, -0.1, -1.0, -0.2, -1.0], dtype=np.float32),
        expert_trajectories=dataset,
        env_name='V2GEnv-discrete',
        cfg=cfg,
    )

    _t0 = time.time()
    trainer.train()
    _elapsed = time.time() - _t0
    print(f"Total training time: {int(_elapsed // 3600)}h {int(_elapsed % 3600 // 60)}m {_elapsed % 60:.1f}s")
