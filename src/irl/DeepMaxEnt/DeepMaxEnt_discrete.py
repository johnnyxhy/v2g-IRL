import numpy as np
import json
import gymnasium as gym
from dataclasses import dataclass
from sbx import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from irl.DeepMaxEnt.DeepMaxEnt import BaseDeepMaxEntTrainer, OBS_SCALES
import os
import torch


# ------------------------------------------------------------------ #
#  Observation wrapper: flatten + normalize Dict obs for MlpPolicy    #
# ------------------------------------------------------------------ #

class FlattenNormalizeObsWrapper(gym.ObservationWrapper):
    """
    Converts the Dict observation space to a normalized flat Box.
    Required because SBX's PPO only supports MlpPolicy (no MultiInputPolicy).
    The normalization matches OBS_SCALES (shared with the discrete env).
    """
    def __init__(self, env):
        super().__init__(env)
        obs_dim = len(OBS_SCALES)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._scales = np.array(OBS_SCALES, dtype=np.float32)

    def observation(self, obs):
        raw = np.concatenate([
            np.asarray(obs['timestep']).flatten(),
            np.asarray(obs['soc']).flatten(),
            np.asarray(obs['soc_gap']).flatten(),
            np.asarray(obs['energy_price']).flatten(),
            np.asarray(obs['battery_capacity']).flatten(),
            np.asarray(obs['time_to_next_journey']).flatten(),
            np.asarray(obs['current_charger_power']).flatten(),
        ]).astype(np.float32)
        return raw / self._scales


# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #

class DeepMaxEntDiscreteConfig:
    """Configuration for Deep MaxEnt IRL training with PPO inner loop (discrete env)."""

    # IRL outer loop
    n_epochs: int = 10
    reward_lr: float = 1e-3
    reward_lr_end: float = 1e-4
    rollout_samples: int = 10

    # PPO inner loop (SBX — does not support clip_range / gae_lambda / device)
    policy_train_steps_per_iter: int = 100_000
    policy_train_lr: float = 3e-4
    policy_gamma: float = 1.0
    policy_batch_size: int = 64
    policy_n_steps: int = 2048
    policy_n_epochs: int = 10
    policy_ent_coef: float = 0.01

    # Data
    train_ratio: float = 0.8
    segment: str = None

    # Reward network architecture
    reward_hidden_dim: int = 64
    reward_obs_dim: int = 7
    reward_action_dim: int = 1

    # Warm-start from a previous run
    pretrained_reward_net_path: str = None

    # Action magnitude penalty
    action_penalty_coeff: float = 0.0

    # Reward scaling for PPO (does not affect IRL gradient)
    reward_scale: float = 1.0

    # Number of parallel envs in DummyVecEnv for PPO training.
    # Profiling shows 4 is the sweet spot (1.82x speedup); 8 gives no further gain.
    n_envs: int = 4

    # Recreate PPO model from scratch at the start of every epoch.
    # Clears Adam momentum (which accumulated for the old reward landscape) so PPO
    # gets a fresh optimisation start each time the reward function changes.
    reset_ppo_each_epoch: bool = True

    # Reward net regularization
    reward_grad_clip: float = 5.0
    reward_l2_reg: float = 0.01

    # Saving
    folder_name: str = "DeepMaxEntIRL_discrete"
    validation: bool = False


# ------------------------------------------------------------------ #
#  Expert data loading                                               #
# ------------------------------------------------------------------ #

@dataclass
class DeepDiscreteExpertTrajectory:
    episodeID: int
    segment: str
    initial_values: dict
    soc_history: list
    observations: np.ndarray      # (n_actions, obs_dim) raw, unnormalized
    actions: np.ndarray           # (n_actions, 1) normalized to [-1, 1]: (discrete-10)/10
    delta_ts: np.ndarray          # (n_actions,) number of env timesteps per action
    feature_expectation: np.ndarray   # (n_features,)


def load_deep_discrete_expert_data(json_path, segment=None, train_ratio=0.8, val_ratio=0.1):
    with open(json_path, 'r') as f:
        data = json.load(f)

    trajectories = []
    for traj in data:
        expert = DeepDiscreteExpertTrajectory(
            episodeID=traj['episodeID'],
            segment=traj['segment'],
            initial_values=traj['initial_values'],
            soc_history=traj['soc_history'],
            observations=np.array(traj['state_action_pairs']['observations'], dtype=np.float32),
            actions=np.array(traj['state_action_pairs']['actions'], dtype=np.float32),
            delta_ts=np.array(traj['state_action_pairs']['delta_ts'], dtype=np.float32),
            feature_expectation=np.array(traj['feature_expectation'], dtype=np.float32),
        )
        trajectories.append(expert)

    if segment is not None:
        trajectories = [t for t in trajectories if segment in t.segment]

    n_train = int(len(trajectories) * train_ratio)
    n_val = int(len(trajectories) * val_ratio)
    return trajectories[:n_train], trajectories[n_train:n_train + n_val], trajectories[n_train + n_val:]


# ------------------------------------------------------------------ #
#  Trainer                                                             #
# ------------------------------------------------------------------ #

class DeepMaxEntDiscreteTrainer(BaseDeepMaxEntTrainer):
    """
    Deep Maximum Entropy IRL with PPO inner-loop for V2GDeepEnv_discrete.

    The reward is a neural network R_θ(s, a) scoring (normalized_obs, action).
    The IRL gradient maximises the likelihood of expert trajectories under the
    MaxEnt distribution:

        ∇θ L = E_expert[∇θ R_θ(s,a)] − E_π[softmax-weighted ∇θ R_θ(s,a)]

    Key differences from the profit SAC variant:
    - PPO inner loop: on-policy, no replay buffer / entropy reset needed.
    - SBX PPO: MlpPolicy only; obs wrapper flattens + normalizes the Dict space.
    """

    plot_title_prefix = "Deep MaxEnt Discrete IRL"

    def __init__(self,
                 train_set: list,
                 val_set: list,
                 env_name: str,
                 cfg: DeepMaxEntDiscreteConfig,
                 test_set: list = None):
        super().__init__(train_set, val_set, env_name, cfg, test_set)
        print(f"Deep MaxEnt Discrete Trainer: {len(self.train_set)} train, {len(self.val_set)} val, {len(self.test_set)} test trajectories.")

    def _process_rollout_action(self, act_arr: np.ndarray) -> torch.Tensor:
        # Rollout acts are raw discrete 0-20; normalize to [-1, 1] to match expert
        return (torch.tensor(act_arr, dtype=torch.float32) - 10.0) / 10.0

    # -------------------------------------------------------------- #
    #  Main training loop                                              #
    # -------------------------------------------------------------- #

    def train(self):
        cfg = self.cfg

        # Create about.md description file
        about_path = f"./models/{cfg.folder_name}/about.md"
        if not os.path.exists(about_path):
            with open(about_path, 'w') as f:
                f.write(cfg.description)

        # Training env — N parallel envs wrapped to flatten+normalize obs for MlpPolicy
        def _make_train_env():
            e = gym.make(self.env_name)
            return FlattenNormalizeObsWrapper(e)

        vec_env = DummyVecEnv([_make_train_env for _ in range(cfg.n_envs)])
        vec_env = VecMonitor(vec_env, filename=f"./models/{cfg.folder_name}/monitor")

        # Separate single env for IRL rollouts — also wrapped so model.predict works
        rollout_env = gym.make(self.env_name)
        rollout_env = FlattenNormalizeObsWrapper(rollout_env)
        rollout_env = DummyVecEnv([lambda: rollout_env])

        def _make_ppo(epoch_idx: int):
            """Create a fresh PPO model. Called once per epoch when reset_ppo_each_epoch=True."""
            return PPO(
                policy="MlpPolicy",
                env=vec_env,
                verbose=0,
                learning_rate=cfg.policy_train_lr,
                gamma=cfg.policy_gamma,
                batch_size=cfg.policy_batch_size,
                n_steps=cfg.policy_n_steps,
                n_epochs=cfg.policy_n_epochs,
                ent_coef=cfg.policy_ent_coef,
                seed=42 + epoch_idx,
                tensorboard_log=f"./models/{cfg.folder_name}/tensorboard/",
            )

        # SBX PPO — MlpPolicy only; clip_range/gae_lambda/device not supported by SBX
        model = _make_ppo(0)

        for epoch in range(cfg.n_epochs):
            print(f"\nEpoch {epoch+1}/{cfg.n_epochs}: Starting training...")

            # ---------- 1. PPO policy training ----------
            if cfg.reset_ppo_each_epoch and epoch > 0:
                model = _make_ppo(epoch)

            for e in vec_env.unwrapped.envs:
                e.unwrapped.set_reward_net(self.reward_net)
                e.unwrapped.set_action_penalty_coeff(cfg.action_penalty_coeff)
                e.unwrapped.set_reward_scale(cfg.reward_scale)
                e.unwrapped.set_initial_states(None)
            rollout_env.envs[0].unwrapped.set_reward_net(self.reward_net)
            rollout_env.envs[0].unwrapped.set_action_penalty_coeff(cfg.action_penalty_coeff)
            rollout_env.envs[0].unwrapped.set_reward_scale(cfg.reward_scale)

            model.learn(
                total_timesteps=cfg.policy_train_steps_per_iter,
                tb_log_name=f"epoch_{epoch+1}",
                progress_bar=True,
                reset_num_timesteps=cfg.reset_ppo_each_epoch,
                log_interval=10,
            )

            # ---------- 2. IRL gradient update ----------
            avg_dtw, avg_mae, avg_feat_l2, avg_reward_loss, avg_log_likelihood = \
                self._irl_gradient_update(rollout_env, model)

            # ---------- 3. Validation ----------
            val_dtw, val_mae, val_feat_l2 = self._run_validation(rollout_env, model)

            # ---------- 4-5. Logging + CSV ----------
            current_lr = self.reward_scheduler.get_last_lr()[0]
            self._log_and_save_epoch(epoch, avg_dtw, avg_mae, avg_feat_l2,
                                     avg_reward_loss, avg_log_likelihood,
                                     val_dtw, val_mae, val_feat_l2, current_lr)

            model.save(f"./models/{cfg.folder_name}/ppo_epoch{epoch+1}")
            torch.save(self.reward_net.state_dict(), f"./models/{cfg.folder_name}/reward_net_epoch{epoch+1}.pt")

        self._plot_results()

        if self.test_set:
            self._evaluate_test_set(rollout_env, model, n_rollouts=30)

        print("Training completed.")

    # -------------------------------------------------------------- #
    #  Helpers                                                         #
    # -------------------------------------------------------------- #

    def _do_rollout(self, rollout_env, model, traj, deterministic=False):
        rollout_env.envs[0].unwrapped.set_initial_states(traj.initial_values)
        obs = rollout_env.reset()

        obs_list, act_list, dt_list = [], [], []
        traj_reward = 0.0

        while True:
            # obs is (1, obs_dim) from DummyVecEnv, already normalized by wrapper
            obs_flat = obs[0].copy()
            action, _ = model.predict(obs, deterministic=deterministic)

            obs_list.append(obs_flat)
            act_list.append(action[0].copy())

            obs, reward, dones, infos = rollout_env.step(action)
            traj_reward += reward[0]
            dt_list.append(infos[0]['delta_t'])

            if bool(dones[0]):
                break

        soc_history = infos[0]['soc_history']
        feature_expectation = np.array(infos[0]['feature_expectation'], dtype=np.float32)
        return obs_list, act_list, dt_list, traj_reward, soc_history, feature_expectation
