"""
Adversarial Inverse Reinforcement Learning (AIRL) for V2G discrete environment.

Based on: Fu, Luo, Levine (2018)
    "Learning Robust Rewards with Adversarial Inverse Reinforcement Learning"
    https://arxiv.org/abs/1710.11248

Architecture
────────────
Generator  — PPO policy π(a|s) trained to maximise the AIRL reward f(s,a,s').

Discriminator — structured sigmoid (NOT a free network):

    D_θ,φ(s,a,s') = σ( f_θ,φ(s,a,s') − log π(a|s) )

    where the reward shaping function is:

        f_θ,φ(s, a, s', Δt, done)
            = g_θ(s,a)                           ← reward net  (interpretable)
            + γ^Δt · (1−done) · h_φ(s')          ← future shaping
            − h_φ(s)                              ← current shaping

    Networks:
        g_θ(s,a)  — reward net:   obs+action → scalar.
                     Disentangled reward we want to recover; at convergence
                     g_θ = r*(s,a) + const regardless of policy.
        h_φ(s)    — shaping net:  obs-only   → scalar.
                     Potential function; ensures Bellman-consistent shaping
                     and removes reward ambiguity from the discriminator.

Discriminator loss (binary cross-entropy, expert=1 / policy=0):

    L_D = −E_expert[log D] − E_π[log(1−D)]
        = BCE( f_expert − log π,  label=1 )
        + BCE( f_policy  − log π,  label=0 )

    Equivalently (numerically stable form using PyTorch BCEWithLogitsLoss):
        logit  = f − log π
        L_D    = BCE_with_logits(logit_expert, ones) + BCE_with_logits(logit_policy, zeros)

PPO reward (generator signal):
    r(s,a,s') = f_θ,φ(s,a,s',Δt,done)           ← no extra Δt scaling;
                                                     γ^Δt handles time discounting

Key differences from DeepMaxEnt IRL
─────────────────────────────────────
    DeepMaxEnt: gradient = φ_expert − φ_agent  (feature matching)
    AIRL:       gradient = ∂L_D/∂θ,φ           (GAN-style discriminator)

    DeepMaxEnt: needs (s, a, Δt)
    AIRL:       needs (s, a, s', done, Δt, log π(a|s))

    DeepMaxEnt: one reward net g_θ
    AIRL:       two nets — g_θ (reward) + h_φ (shaping)

Variable Δt
───────────
The environment uses variable-length actions spanning Δt physical timesteps.
The shaping term uses γ^Δt (matching the VariableDt Bellman backup) so that
the discriminator remains consistent with the PPO discount.
"""

import warnings
import numpy as np
import json
import csv
import gymnasium as gym
from dataclasses import dataclass
from sbx import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecEnvWrapper
from irl.DeepMaxEnt.DeepMaxEnt import RewardNet, PROFIT_OBS_SCALES
from irl.utils.tools import compute_dtw
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Observation wrapper: flatten + normalize Dict obs for MlpPolicy    #
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
#  AIRL reward callback: r = f - log π (applied after rollout)        #
# ------------------------------------------------------------------ #

class AIRLRewardVecEnvWrapper(VecEnvWrapper):
    """
    Intercepts env step rewards and subtracts log π(a|s) inline, giving PPO
    the correct AIRL reward at the point of collection:

        r(s, a, s') = f(s, a, s') − log π(a|s)  =  log D − log(1−D)

    The underlying env already computes f(s,a,s') as its step reward.
    This wrapper adds the −log π correction using SBX's JAX policy API.

    Must call set_model(model) before each model.learn() call.
    """

    def __init__(self, venv):
        super().__init__(venv)
        self.model = None
        self._last_obs = None

    def set_model(self, model) -> None:
        import jax
        import jax.numpy as jnp
        self.model = model
        # JIT-compile the log_prob lookup once per model (params are args, not closures)
        apply_fn = model.policy.actor_state.apply_fn
        @jax.jit
        def _log_prob(params, obs, acts):
            dist = apply_fn(params, obs)
            return dist.log_prob(acts)
        self._log_prob_jit = _log_prob

    def reset(self):
        obs = self.venv.reset()
        self._last_obs = obs.copy()
        return obs

    def step_async(self, actions):
        self._last_actions = actions.copy()
        self.venv.step_async(actions)

    def step_wait(self):
        import jax.numpy as jnp
        obs, rewards, dones, infos = self.venv.step_wait()
        if self.model is not None and self._last_obs is not None:
            obs_jax = jnp.array(self._last_obs)                          # (n_envs, obs_dim)
            act_jax = jnp.array(self._last_actions, dtype=jnp.int32)    # (n_envs,)
            log_pi = np.array(
                self._log_prob_jit(self.model.policy.actor_state.params, obs_jax, act_jax),
                dtype=np.float32
            )  # (n_envs,)
            rewards = rewards - log_pi
        self._last_obs = obs.copy()
        return obs, rewards, dones, infos


class FlattenNormalizeObsWrapper(gym.ObservationWrapper):
    """
    Converts the Dict observation space to a normalized flat Box.
    Normalization uses PROFIT_OBS_SCALES (shared with the env).
    """
    def __init__(self, env):
        super().__init__(env)
        obs_dim = len(PROFIT_OBS_SCALES)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._scales = np.array(PROFIT_OBS_SCALES, dtype=np.float32)

    def observation(self, obs):
        raw = np.concatenate([
            np.asarray(obs['timestep']).flatten(),
            np.asarray(obs['soc']).flatten(),
            np.asarray(obs['soc_target']).flatten(),
            np.asarray(obs['energy_price']).flatten(),
            np.asarray(obs['battery_capacity']).flatten(),
            np.asarray(obs['time_to_next_journey']).flatten(),
            np.asarray(obs['current_charger_power']).flatten(),
        ]).astype(np.float32)
        return raw / self._scales


# ------------------------------------------------------------------ #
#  Shaping network h_φ(s)                                             #
# ------------------------------------------------------------------ #

class ShapingNet(nn.Module):
    """
    Neural network approximating the AIRL potential function h_φ(s).

    Takes only the (normalized) observation as input — no action — and
    outputs an unbounded scalar. At convergence the shaping term
        γ^Δt · h_φ(s') − h_φ(s)
    cancels out of the recovered reward g_θ(s,a), leaving g_θ = r* + const.

    Architecture mirrors RewardNet (two hidden layers, ReLU, linear out) but
    is narrower because state-only inputs carry less information.
    """

    def __init__(self, obs_dim: int = 7, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim) normalized observations
        Returns:
            (batch,) scalar potential values
        """
        return self.net(obs).squeeze(-1)


# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #

class AIRLConfig:
    """Configuration for AIRL training with PPO inner loop (discrete env)."""

    # IRL outer loop
    n_epochs: int = 10
    disc_lr: float = 1e-3               # discriminator (g_θ + h_φ) learning rate
    disc_lr_end: float = 1e-4           # final LR after cosine annealing
    rollout_samples: int = 10           # policy rollouts per expert trajectory for discriminator

    # Discriminator network dimensions
    reward_hidden_dim: int = 64
    shaping_hidden_dim: int = 64
    reward_obs_dim: int = 7
    reward_action_dim: int = 1

    # Warm-start from a previous run
    pretrained_reward_net_path: str = None
    pretrained_shaping_net_path: str = None

    # PPO inner loop (SBX — does not support clip_range / gae_lambda / device)
    policy_train_steps_per_iter: int = 100_000
    policy_train_lr: float = 3e-4
    # γ=1.0 for consistency with variable-Δt actions: SBX PPO applies its discount
    # once per action step, so any γ<1 would mismatch the γ^Δt shaping term in
    # _compute_f.  Undiscounted returns are appropriate for this finite-horizon task.
    policy_gamma: float = 1.0
    policy_batch_size: int = 64
    policy_n_steps: int = 2048
    policy_n_epochs: int = 10
    policy_ent_coef: float = 0.01

    # Data
    train_ratio: float = 0.8
    segment: str = None

    # Discriminator training epochs per IRL epoch.
    # Rollouts are collected once; the discriminator then runs disc_epochs passes,
    # stepping after EACH trajectory (mini-batch GD).  Total updates per IRL epoch:
    # disc_epochs × N_train (e.g. 10 × 75 = 750).  Increase if disc_loss stays high.
    disc_epochs: int = 10

    # Discriminator regularisation
    disc_l2_reg: float = 0.01

    # Action magnitude penalty for PPO (does not affect discriminator gradient)
    action_penalty_coeff: float = 0.0

    # Reward scaling for PPO (does not affect discriminator gradient)
    reward_scale: float = 1.0

    # Behavioral Cloning pre-training (optional).
    # Trains the PPO actor to mimic expert actions before AIRL starts, bootstrapping
    # the policy into the expert distribution so the discriminator provides a
    # meaningful gradient signal from epoch 1 instead of teaching from scratch.
    bc_pretrain_steps: int = 0   # number of Adam gradient steps (0 = disabled)
    bc_lr: float = 1e-3          # BC Adam learning rate

    # Number of parallel envs in DummyVecEnv for PPO training.
    n_envs: int = 4

    # Reset PPO optimiser each epoch so stale momentum from the old reward
    # landscape does not bias the new policy.
    reset_ppo_each_epoch: bool = True

    # Run validation rollouts on the held-out set each epoch.
    validation: bool = False

    # Human-readable description written to about.md in the model folder.
    description: str = ""

    # Saving
    folder_name: str = "AIRL_discrete"


# ------------------------------------------------------------------ #
#  Expert data loading (AIRL format)                                   #
# ------------------------------------------------------------------ #

@dataclass
class AIRLExpertTrajectory:
    """
    Expert demonstration in AIRL format.

    The three fields absent from the DeepMaxEnt format are required to
    evaluate the discriminator:

        next_observations  — s' for each action (to compute h_φ(s'))
        dones              — terminal flag (masks γ^Δt·h_φ(s') at episode end)
        raw_actions        — discrete integer [0,20] for log π(a|s) with Categorical PPO
    """
    episodeID: int
    segment: str
    initial_values: dict
    soc_history: list
    observations: np.ndarray        # (N, obs_dim) raw, unnormalized  s
    next_observations: np.ndarray   # (N, obs_dim) raw, unnormalized  s'
    actions: np.ndarray             # (N, 1) normalized [-1,1]  for g_θ input
    raw_actions: np.ndarray         # (N, 1) int [0,20]          for log π(a|s)
    delta_ts: np.ndarray            # (N,) number of env timesteps per action
    dones: np.ndarray               # (N,) bool: True only for terminal action
    feature_expectation: np.ndarray # (n_features,) for monitoring only


def load_airl_expert_data(json_path: str, segment=None, train_ratio: float = 0.8):
    """
    Load AIRL-format trajectories from a JSON file produced by
    expert_loader_airl_discrete.py.

    Returns:
        train_set, val_set — lists of AIRLExpertTrajectory
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    trajectories = []
    for traj in data:
        sap = traj['state_action_pairs']
        expert = AIRLExpertTrajectory(
            episodeID=traj['episodeID'],
            segment=traj['segment'],
            initial_values=traj['initial_values'],
            soc_history=traj['soc_history'],
            observations=np.array(sap['observations'],      dtype=np.float32),
            next_observations=np.array(sap['next_observations'], dtype=np.float32),
            actions=np.array(sap['actions'],                dtype=np.float32),
            raw_actions=np.array(sap['raw_actions'],        dtype=np.int64),
            delta_ts=np.array(sap['delta_ts'],              dtype=np.float32),
            dones=np.array(sap['dones'],                    dtype=np.float32),
            feature_expectation=np.array(traj['feature_expectation'], dtype=np.float32),
        )
        trajectories.append(expert)

    if segment is not None:
        trajectories = [t for t in trajectories if segment in t.segment]

    n_train = int(len(trajectories) * train_ratio)
    return trajectories[:n_train], trajectories[n_train:]


# ------------------------------------------------------------------ #
#  AIRL Trainer                                                        #
# ------------------------------------------------------------------ #

class AIRLTrainer:
    """
    Adversarial IRL trainer for V2GDeepEnv_discrete using SBX PPO.

    Outer loop (per epoch):
        1. Set g_θ and h_φ in all training envs.
        2. Train PPO for `policy_train_steps_per_iter` steps under the current
           AIRL reward  f = g_θ(s,a) + γ^Δt·(1−done)·h_φ(s') − h_φ(s).
        3. For each expert trajectory:
             a. Collect `rollout_samples` stochastic policy rollouts from the
                same initial state as the expert.
             b. Compute log π(a|s) for expert transitions (current policy).
             c. Compute log π(a|s) for all policy rollout transitions.
             d. Compute discriminator logits for expert and policy:
                    logit = f(s,a,s',Δt,done) − log π(a|s)
             e. Accumulate binary cross-entropy loss (expert=1, policy=0).
        4. Clip gradients and update g_θ, h_φ with AdamW + cosine LR decay.
        5. (Optional) validate on held-out trajectories.

    Convergence property (Fu et al. 2018):
        At convergence, g_θ(s,a) = r*(s,a) + const.  Unlike GAIL, the reward
        is disentangled from the policy and transfers to new environments.
    """

    def __init__(self,
                 train_set: list,
                 val_set: list,
                 env_name: str,
                 cfg: AIRLConfig):
        self.cfg = cfg
        self.env_name = env_name
        self.train_set = train_set
        self.val_set = val_set

        self.obs_scales = torch.tensor(PROFIT_OBS_SCALES, dtype=torch.float32)
        self.device = torch.device("cpu")

        # ---- Networks ----
        self.reward_net = RewardNet(
            obs_dim=cfg.reward_obs_dim,
            action_dim=cfg.reward_action_dim,
            hidden_dim=cfg.reward_hidden_dim,
        )
        self.shaping_net = ShapingNet(
            obs_dim=cfg.reward_obs_dim,
            hidden_dim=cfg.shaping_hidden_dim,
        )

        if cfg.pretrained_reward_net_path is not None:
            self.reward_net.load_state_dict(
                torch.load(cfg.pretrained_reward_net_path, weights_only=True)
            )
            print(f"Loaded pretrained reward net from {cfg.pretrained_reward_net_path}")

        if cfg.pretrained_shaping_net_path is not None:
            self.shaping_net.load_state_dict(
                torch.load(cfg.pretrained_shaping_net_path, weights_only=True)
            )
            print(f"Loaded pretrained shaping net from {cfg.pretrained_shaping_net_path}")

        # ---- Single optimiser for both discriminator networks ----
        # Using AdamW weight_decay as L2 regularization on both g_θ and h_φ.
        self.disc_optimizer = torch.optim.AdamW(
            list(self.reward_net.parameters()) + list(self.shaping_net.parameters()),
            lr=cfg.disc_lr,
            weight_decay=cfg.disc_l2_reg,
        )
        self.disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.disc_optimizer, T_max=cfg.n_epochs, eta_min=cfg.disc_lr_end
        )

        # ---- Tracking ----
        self.train_dtw_distance = []
        self.train_mae = []
        self.train_feat_l2 = []
        self.val_dtw_distance = []
        self.val_mae = []
        self.val_feat_l2 = []
        self.train_disc_loss = []           # discriminator BCE loss
        self.train_expert_acc = []          # fraction of expert transitions classified as expert
        self.train_policy_acc = []          # fraction of policy transitions classified as policy

        os.makedirs(f"./models/{cfg.folder_name}", exist_ok=True)
        print(
            f"AIRL Trainer: {len(self.train_set)} train, {len(self.val_set)} val trajectories."
        )

    # -------------------------------------------------------------- #
    #  Main training loop                                              #
    # -------------------------------------------------------------- #

    def train(self):
        cfg = self.cfg

        about_path = f"./models/{cfg.folder_name}/about.md"
        if not os.path.exists(about_path) and cfg.description:
            with open(about_path, 'w', encoding='utf-8') as f:
                f.write(cfg.description)

        def _make_train_env():
            e = gym.make(self.env_name)
            return FlattenNormalizeObsWrapper(e)

        vec_env = DummyVecEnv([_make_train_env for _ in range(cfg.n_envs)])
        vec_env = VecMonitor(vec_env, filename=f"./models/{cfg.folder_name}/monitor")
        # Wrap with AIRL reward correction: r = f - log π applied at collection time.
        # set_model() must be called before each model.learn().
        airl_wrapper = AIRLRewardVecEnvWrapper(vec_env)

        rollout_base_env = gym.make(self.env_name)
        rollout_base_env = FlattenNormalizeObsWrapper(rollout_base_env)
        rollout_env = DummyVecEnv([lambda: rollout_base_env])

        def _make_ppo(epoch_idx: int):
            return PPO(
                policy="MlpPolicy",
                env=airl_wrapper,
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

        model = _make_ppo(0)

        # Optional BC pre-training: initialise policy near expert distribution
        # before the AIRL outer loop begins.
        if cfg.bc_pretrain_steps > 0:
            self._bc_pretrain(model)

        for epoch in range(cfg.n_epochs):
            print(f"\nEpoch {epoch+1}/{cfg.n_epochs}")

            # ---- 1. PPO training under current AIRL reward ----
            if cfg.reset_ppo_each_epoch and epoch > 0:
                model = _make_ppo(epoch)

            # Push current g_θ and h_φ into all envs before PPO training.
            for e in vec_env.unwrapped.envs:
                e.unwrapped.set_reward_net(self.reward_net)
                e.unwrapped.set_shaping_net(self.shaping_net)
                e.unwrapped.set_gamma(cfg.policy_gamma)
                e.unwrapped.set_action_penalty_coeff(cfg.action_penalty_coeff)
                e.unwrapped.set_reward_scale(cfg.reward_scale)
                e.unwrapped.set_initial_states(None)
            rollout_env.envs[0].unwrapped.set_reward_net(self.reward_net)
            rollout_env.envs[0].unwrapped.set_shaping_net(self.shaping_net)
            rollout_env.envs[0].unwrapped.set_gamma(cfg.policy_gamma)
            rollout_env.envs[0].unwrapped.set_action_penalty_coeff(cfg.action_penalty_coeff)
            rollout_env.envs[0].unwrapped.set_reward_scale(cfg.reward_scale)

            # Give the reward wrapper a reference to the current policy so it can
            # compute log π(a|s) during rollout collection.
            airl_wrapper.set_model(model)

            model.learn(
                total_timesteps=cfg.policy_train_steps_per_iter,
                tb_log_name=f"epoch_{epoch+1}",
                progress_bar=True,
                reset_num_timesteps=cfg.reset_ppo_each_epoch,
                log_interval=10,
            )

            # ---- 2. Discriminator update ----
            # Phase A — collect all transitions once (rollouts are expensive).
            # log_pi is cached here because the policy doesn't change during disc training.
            # f is NOT cached; it is recomputed each disc epoch as g_θ/h_φ weights update.
            N = len(self.train_set)
            obs_scales = self.obs_scales

            avg_dtw     = 0.0
            avg_mae     = 0.0
            avg_feat_l2 = 0.0

            collected = []  # (expert_tensors..., log_pi_e, policy_tensors..., log_pi_p)
            for traj in tqdm(self.train_set, desc="Collecting rollouts"):
                expert_obs      = torch.tensor(traj.observations,      dtype=torch.float32) / obs_scales
                expert_next_obs = torch.tensor(traj.next_observations, dtype=torch.float32) / obs_scales
                expert_act_norm = torch.tensor(traj.actions,           dtype=torch.float32)  # (T,1) normalized
                expert_raw_act  = traj.raw_actions.flatten()           # (T,) int [0,20]
                expert_dt       = torch.tensor(traj.delta_ts,          dtype=torch.float32)  # (T,)
                expert_dones    = torch.tensor(traj.dones,             dtype=torch.float32)  # (T,)

                log_pi_expert = torch.tensor(
                    self._get_log_prob(model, expert_obs.numpy(), expert_raw_act),
                    dtype=torch.float32,
                )  # (T,)

                policy_obs_all      = []
                policy_next_obs_all = []
                policy_act_norm_all = []
                policy_raw_act_all  = []
                policy_dt_all       = []
                policy_done_all     = []

                for _ in range(cfg.rollout_samples):
                    (obs_list, raw_acts, next_obs_list, dones_list,
                     act_norm_list, dt_list,
                     _, soc_hist, feat_exp) = self._do_rollout(rollout_env, model, traj)

                    policy_obs_all.extend(obs_list)
                    policy_raw_act_all.extend(raw_acts)
                    policy_next_obs_all.extend(next_obs_list)
                    policy_done_all.extend(dones_list)
                    policy_act_norm_all.extend(act_norm_list)
                    policy_dt_all.extend(dt_list)

                    expert_soc = np.array(traj.soc_history, dtype=np.float32)
                    avg_dtw += compute_dtw(expert_soc, np.array(soc_hist, dtype=np.float32)) / (
                        N * cfg.rollout_samples
                    )
                    avg_mae += self._compute_mae(expert_soc, np.array(soc_hist, dtype=np.float32)) / (
                        N * cfg.rollout_samples
                    )
                    avg_feat_l2 += np.linalg.norm(traj.feature_expectation - feat_exp) / (
                        N * cfg.rollout_samples
                    )

                policy_obs_t      = torch.tensor(np.array(policy_obs_all),      dtype=torch.float32)
                policy_next_obs_t = torch.tensor(np.array(policy_next_obs_all), dtype=torch.float32)
                policy_act_norm_t = torch.tensor(np.array(policy_act_norm_all), dtype=torch.float32).unsqueeze(-1)
                policy_raw_act_np = np.array(policy_raw_act_all, dtype=np.int64)
                policy_dt_t       = torch.tensor(np.array(policy_dt_all),       dtype=torch.float32)
                policy_done_t     = torch.tensor(np.array(policy_done_all),     dtype=torch.float32)

                log_pi_policy = torch.tensor(
                    self._get_log_prob(model, policy_obs_t.numpy(), policy_raw_act_np),
                    dtype=torch.float32,
                )  # (B,)

                collected.append((
                    expert_obs, expert_next_obs, expert_act_norm, expert_dt, expert_dones, log_pi_expert,
                    policy_obs_t, policy_next_obs_t, policy_act_norm_t, policy_dt_t, policy_done_t, log_pi_policy,
                ))

            # Phase B — train discriminator for disc_epochs passes on the cached data.
            # f is recomputed each pass (g_θ/h_φ weights change); log_pi is reused.
            avg_disc_loss  = 0.0
            avg_expert_acc = 0.0
            avg_policy_acc = 0.0

            # Diagnostic values — populated on the last disc epoch
            diag_f_expert_mean   = 0.0
            diag_f_policy_mean   = 0.0
            diag_g_expert_mean   = 0.0
            diag_g_policy_mean   = 0.0
            diag_h_expert_mean   = 0.0
            diag_h_policy_mean   = 0.0
            diag_logpi_e_mean    = 0.0
            diag_logpi_p_mean    = 0.0
            diag_logit_e_mean    = 0.0
            diag_logit_p_mean    = 0.0
            total_grad_norm      = 0.0

            # Mini-batch GD: step after EACH trajectory, not once after accumulating
            # all N.  This gives disc_epochs × N gradient updates per IRL epoch
            # instead of disc_epochs × 1, which is essential for the discriminator
            # to converge (f needs to move by ~3 units; one full-batch step moves it
            # by ~0.001).
            for disc_ep in tqdm(range(cfg.disc_epochs), desc="Disc epochs"):
                ep_loss       = 0.0
                ep_expert_acc = 0.0
                ep_policy_acc = 0.0
                is_last = (disc_ep == cfg.disc_epochs - 1)

                for (expert_obs, expert_next_obs, expert_act_norm,
                     expert_dt, expert_dones, log_pi_expert,
                     policy_obs_t, policy_next_obs_t, policy_act_norm_t,
                     policy_dt_t, policy_done_t, log_pi_policy) in collected:

                    self.disc_optimizer.zero_grad()

                    f_expert = self._compute_f(
                        expert_obs, expert_act_norm,
                        expert_next_obs, expert_dones, expert_dt,
                    )  # (T,)
                    f_policy = self._compute_f(
                        policy_obs_t, policy_act_norm_t,
                        policy_next_obs_t, policy_done_t, policy_dt_t,
                    )  # (B,)

                    logit_expert = f_expert - log_pi_expert  # (T,)
                    logit_policy = f_policy - log_pi_policy  # (B,)

                    loss_expert = F.binary_cross_entropy_with_logits(
                        logit_expert, torch.ones_like(logit_expert), reduction='mean',
                    )
                    loss_policy = F.binary_cross_entropy_with_logits(
                        logit_policy, torch.zeros_like(logit_policy), reduction='mean',
                    )
                    # No /N: we step after each trajectory so losses don't accumulate
                    loss_traj = (loss_expert + loss_policy) / 2
                    loss_traj.backward()

                    # Grad norm — tracked on last trajectory of last disc epoch
                    if is_last:
                        gn = 0.0
                        for p in (list(self.reward_net.parameters())
                                  + list(self.shaping_net.parameters())):
                            if p.grad is not None:
                                gn += p.grad.data.norm(2).item() ** 2
                        total_grad_norm = gn ** 0.5

                    self.disc_optimizer.step()

                    ep_loss += loss_traj.item() / N
                    with torch.no_grad():
                        ep_expert_acc += (logit_expert > 0).float().mean().item() / N
                        ep_policy_acc += (logit_policy < 0).float().mean().item() / N

                    # Diagnostics — only on the last disc epoch to avoid overhead
                    if is_last:
                        with torch.no_grad():
                            diag_f_expert_mean  += f_expert.detach().mean().item() / N
                            diag_f_policy_mean  += f_policy.detach().mean().item() / N
                            diag_g_expert_mean  += self.reward_net(expert_obs, expert_act_norm).mean().item() / N
                            diag_g_policy_mean  += self.reward_net(policy_obs_t, policy_act_norm_t).mean().item() / N
                            diag_h_expert_mean  += self.shaping_net(expert_obs).mean().item() / N
                            diag_h_policy_mean  += self.shaping_net(policy_obs_t).mean().item() / N
                            diag_logpi_e_mean   += log_pi_expert.mean().item() / N
                            diag_logpi_p_mean   += log_pi_policy.mean().item() / N
                            diag_logit_e_mean   += logit_expert.detach().mean().item() / N
                            diag_logit_p_mean   += logit_policy.detach().mean().item() / N

                avg_disc_loss  += ep_loss        / cfg.disc_epochs
                avg_expert_acc += ep_expert_acc  / cfg.disc_epochs
                avg_policy_acc += ep_policy_acc  / cfg.disc_epochs

            # Scheduler steps once per IRL epoch (not per disc epoch) so cosine
            # annealing tracks n_epochs correctly regardless of disc_epochs.
            self.disc_scheduler.step()
            current_lr = self.disc_scheduler.get_last_lr()[0]

            # ---- 3. Validation ----
            val_dtw = 0.0
            val_mae = 0.0
            val_feat_l2 = 0.0
            if cfg.validation and len(self.val_set) > 0:
                M = len(self.val_set)
                for traj in tqdm(self.val_set, desc="Validation"):
                    (_, _, _, _, _, _, _, soc_hist, feat_exp) = self._do_rollout(
                        rollout_env, model, traj, deterministic=False
                    )
                    val_dtw += compute_dtw(
                        np.array(traj.soc_history, dtype=np.float32),
                        np.array(soc_hist, dtype=np.float32),
                    ) / M
                    val_mae += self._compute_mae(
                        np.array(traj.soc_history, dtype=np.float32),
                        np.array(soc_hist, dtype=np.float32),
                    ) / M
                    val_feat_l2 += np.linalg.norm(traj.feature_expectation - feat_exp) / M

            # ---- 4. Logging ----
            self.train_dtw_distance.append(avg_dtw)
            self.train_mae.append(avg_mae)
            self.train_feat_l2.append(avg_feat_l2)
            self.train_disc_loss.append(avg_disc_loss)
            self.train_expert_acc.append(avg_expert_acc)
            self.train_policy_acc.append(avg_policy_acc)
            if cfg.validation and len(self.val_set) > 0:
                self.val_dtw_distance.append(val_dtw)
                self.val_mae.append(val_mae)
                self.val_feat_l2.append(val_feat_l2)

            print(f"--- Epoch {epoch+1}/{cfg.n_epochs} Summary ---")
            print(
                f"Disc Loss: {avg_disc_loss:.4f}  "
                f"Expert Acc: {avg_expert_acc:.3f}  Policy Acc: {avg_policy_acc:.3f}  "
                f"LR: {current_lr:.6f}"
            )
            print(
                f"  f:       expert={diag_f_expert_mean:+.4f}   policy={diag_f_policy_mean:+.4f}"
                f"   gap={diag_f_expert_mean - diag_f_policy_mean:+.4f}"
            )
            print(
                f"  g_theta: expert={diag_g_expert_mean:+.4f}   policy={diag_g_policy_mean:+.4f}"
            )
            print(
                f"  h_phi:   expert={diag_h_expert_mean:+.4f}   policy={diag_h_policy_mean:+.4f}"
            )
            print(
                f"  log_pi:  expert={diag_logpi_e_mean:+.4f}   policy={diag_logpi_p_mean:+.4f}"
            )
            print(
                f"  logit:   expert={diag_logit_e_mean:+.4f}   policy={diag_logit_p_mean:+.4f}"
                f"   (need expert>0, policy<0)"
            )
            print(f"  grad_norm: {total_grad_norm:.4f}")
            print(f"Train DTW: {avg_dtw:.4f}  Train MAE: {avg_mae:.4f}  Train Feat L2: {avg_feat_l2:.4f}")
            if cfg.validation and len(self.val_set) > 0:
                print(f"Val DTW: {val_dtw:.4f}  Val MAE: {val_mae:.4f}  Val Feat L2: {val_feat_l2:.4f}")

            # ---- 5. Save metrics (append — survives checkpoint continuations) ----
            metrics_path = f"./models/{cfg.folder_name}/metrics.csv"
            write_header = not os.path.exists(metrics_path)
            with open(metrics_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        'epoch', 'train_dtw', 'train_mae', 'train_feat_l2',
                        'disc_loss', 'expert_acc', 'policy_acc',
                        'val_dtw', 'val_mae', 'val_feat_l2', 'lr',
                    ])
                writer.writerow([
                    epoch + 1,
                    round(avg_dtw, 6),
                    round(avg_mae, 6),
                    round(avg_feat_l2, 6),
                    round(avg_disc_loss, 6),
                    round(avg_expert_acc, 4),
                    round(avg_policy_acc, 4),
                    round(val_dtw, 6) if (cfg.validation and len(self.val_set) > 0) else '',
                    round(val_mae, 6) if (cfg.validation and len(self.val_set) > 0) else '',
                    round(val_feat_l2, 6) if (cfg.validation and len(self.val_set) > 0) else '',
                    round(current_lr, 8),
                ])

            model.save(f"./models/{cfg.folder_name}/ppo_epoch{epoch+1}")
            torch.save(
                self.reward_net.state_dict(),
                f"./models/{cfg.folder_name}/reward_net_epoch{epoch+1}.pt",
            )
            torch.save(
                self.shaping_net.state_dict(),
                f"./models/{cfg.folder_name}/shaping_net_epoch{epoch+1}.pt",
            )

        self._plot_results()
        print("AIRL training completed.")

    # -------------------------------------------------------------- #
    #  Discriminator helpers                                           #
    # -------------------------------------------------------------- #

    def _compute_f(
        self,
        obs: torch.Tensor,       # (B, obs_dim) normalized
        act_norm: torch.Tensor,  # (B, 1) or (B,) normalized [-1,1]
        next_obs: torch.Tensor,  # (B, obs_dim) normalized
        dones: torch.Tensor,     # (B,) float 0/1
        delta_ts: torch.Tensor,  # (B,) float
    ) -> torch.Tensor:
        """
        Compute the AIRL reward shaping function:

            f(s, a, s', Δt, done)
                = g_θ(s, a)
                + γ^Δt · (1 − done) · h_φ(s')
                − h_φ(s)

        This is the quantity that enters the discriminator logit alongside
        −log π(a|s).  It is differentiable w.r.t. θ and φ.
        """
        if act_norm.dim() == 1:
            act_norm = act_norm.unsqueeze(-1)

        g   = self.reward_net(obs, act_norm)           # (B,)
        h_s = self.shaping_net(obs)                    # (B,)
        h_sp = self.shaping_net(next_obs)              # (B,)
        gamma_dt = self.cfg.policy_gamma ** delta_ts   # (B,) element-wise

        return g + gamma_dt * (1.0 - dones) * h_sp - h_s  # (B,)

    def _bc_pretrain(self, model: PPO) -> None:
        """
        Behavioural Cloning pre-training for the SBX PPO actor via optax.

        Maximises  E[log π(a_expert | s_expert)]  on the training set, placing
        the policy near the expert's action distribution before AIRL starts.
        This prevents the discriminator from diverging on a near-random policy
        and gives the AIRL reward gradient a meaningful starting point.

        Only the actor params are updated; the critic is untouched.  The
        actor's PPO optimizer state is reset afterwards so stale BC momentum
        does not bias the first AIRL epoch.
        """
        import jax
        import jax.numpy as jnp
        import optax

        cfg = self.cfg
        obs_scale = np.array(PROFIT_OBS_SCALES, dtype=np.float32)

        # Flatten all expert (obs, raw_action) pairs from the training set.
        all_obs = np.concatenate([
            t.observations / obs_scale for t in self.train_set
        ])  # (N_total, obs_dim)
        all_acts = np.concatenate([
            t.raw_actions.flatten() for t in self.train_set
        ]).astype(np.int32)  # (N_total,)

        apply_fn  = model.policy.actor_state.apply_fn
        bc_optim  = optax.adam(cfg.bc_lr)
        bc_opt_st = bc_optim.init(model.policy.actor_state.params)

        @jax.jit
        def _step(params, opt_state, obs_b, act_b):
            def loss_fn(p):
                return -apply_fn(p, obs_b).log_prob(act_b).mean()
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt = bc_optim.update(grads, opt_state)
            return optax.apply_updates(params, updates), new_opt, loss

        n = len(all_obs)
        batch_size = 256
        print(f"BC pre-training: {cfg.bc_pretrain_steps} steps on {n} expert transitions")

        for step in range(cfg.bc_pretrain_steps):
            idx = np.random.randint(0, n, batch_size)
            new_params, bc_opt_st, loss = _step(
                model.policy.actor_state.params,
                bc_opt_st,
                jnp.array(all_obs[idx]),
                jnp.array(all_acts[idx]),
            )
            model.policy.actor_state = model.policy.actor_state.replace(params=new_params)

            if (step + 1) % 1000 == 0:
                print(f"  BC step {step+1}/{cfg.bc_pretrain_steps}: loss={float(loss):.4f}")

        # Reset the PPO actor optimizer state so stale BC momentum doesn't pollute
        # the first AIRL epoch's PPO updates.
        try:
            fresh_opt_state = model.policy.actor_state.tx.init(
                model.policy.actor_state.params
            )
            model.policy.actor_state = model.policy.actor_state.replace(
                opt_state=fresh_opt_state
            )
        except AttributeError:
            pass  # Older SBX — live with stale momentum; impact is minor.

        print("BC pre-training done.")

    def _get_log_prob(
        self,
        model: PPO,
        obs_np: np.ndarray,        # (B, obs_dim) float32, normalized
        raw_actions_np: np.ndarray, # (B,) int64, discrete [0, 20]
    ) -> np.ndarray:
        """
        Evaluate log π(a|s) using the current PPO Categorical policy.

        SBX PPO is JAX-accelerated but follows SB3's ActorCriticPolicy interface.
        `evaluate_actions(obs, actions)` returns (values, log_prob, entropy).

        Args:
            obs_np:        (B, obs_dim) float32 — already normalized by PROFIT_OBS_SCALES
            raw_actions_np: (B,) int64  — discrete action index [0, 20]

        Returns:
            log_probs: (B,) float32 numpy array
        """
        # SBX uses JAX: apply_fn returns a distrax Categorical distribution.
        import jax.numpy as jnp
        obs_jax = jnp.array(obs_np)
        act_jax = jnp.array(raw_actions_np, dtype=jnp.int32)
        dist = model.policy.actor_state.apply_fn(
            model.policy.actor_state.params, obs_jax
        )
        return np.array(dist.log_prob(act_jax), dtype=np.float32)

    # -------------------------------------------------------------- #
    #  Rollout helper                                                  #
    # -------------------------------------------------------------- #

    def _do_rollout(self, rollout_env, model, traj, deterministic=False):
        """
        Run one episode from the expert's initial state.

        Returns a tuple containing all fields needed for the AIRL discriminator:
            obs_list       — (T, obs_dim) float32 normalized  s
            raw_act_list   — (T,) int64             a (discrete [0,20])
            next_obs_list  — (T, obs_dim) float32 normalized  s'
            dones_list     — (T,) float   done flag
            act_norm_list  — (T,) float   (a − 10)/10 normalized for reward net
            dt_list        — (T,) int     Δt per action
            traj_reward    — float        cumulative env reward (for monitoring)
            soc_history    — list[float]  SoC at each physical timestep
            feature_expectation — (n_features,) np.float32
        """
        rollout_env.envs[0].unwrapped.set_initial_states(traj.initial_values)
        obs = rollout_env.reset()

        obs_list      = []
        raw_act_list  = []
        next_obs_list = []
        dones_list    = []
        act_norm_list = []
        dt_list       = []
        traj_reward   = 0.0

        while True:
            obs_flat = obs[0].copy()  # (obs_dim,) normalized — this is s
            action, _ = model.predict(obs, deterministic=deterministic)
            raw_action = int(action[0])  # discrete integer [0, 20]

            next_obs, reward, dones_arr, infos = rollout_env.step(action)
            done = bool(dones_arr[0])

            obs_list.append(obs_flat)
            raw_act_list.append(raw_action)
            next_obs_list.append(next_obs[0].copy())  # s' (normalized by wrapper)
            dones_list.append(float(done))
            act_norm_list.append((raw_action - 10.0) / 10.0)
            dt_list.append(infos[0]['delta_t'])
            traj_reward += float(reward[0])

            obs = next_obs
            if done:
                break

        soc_history = infos[0]['soc_history']
        feature_expectation = np.array(infos[0]['feature_expectation'], dtype=np.float32)

        return (
            obs_list, raw_act_list, next_obs_list, dones_list,
            act_norm_list, dt_list,
            traj_reward, soc_history, feature_expectation,
        )

    # -------------------------------------------------------------- #
    #  Plotting                                                        #
    # -------------------------------------------------------------- #

    @staticmethod
    def _compute_mae(soc_a: np.ndarray, soc_b: np.ndarray) -> float:
        """MAE between two SoC trajectories, resampling to the longer length."""
        n = max(len(soc_a), len(soc_b))
        if len(soc_a) != n:
            soc_a = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(soc_a)), soc_a)
        if len(soc_b) != n:
            soc_b = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(soc_b)), soc_b)
        return float(np.mean(np.abs(soc_a - soc_b)))

    def _plot_results(self):
        cfg = self.cfg
        epochs = range(1, cfg.n_epochs + 1)

        plt.figure(1)
        plt.plot(epochs, self.train_dtw_distance, label='Train DTW')
        if cfg.validation and len(self.val_set) > 0:
            plt.plot(epochs, self.val_dtw_distance, label='Val DTW')
            plt.legend()
        plt.title('AIRL — DTW Distance')
        plt.xlabel('Epoch'); plt.ylabel('Average DTW Distance'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/dtw_distance.png')

        plt.figure(2)
        plt.plot(epochs, self.train_mae, label='Train MAE')
        if cfg.validation and len(self.val_set) > 0 and self.val_mae:
            plt.plot(epochs, self.val_mae, label='Val MAE')
            plt.legend()
        plt.title('AIRL — SoC MAE')
        plt.xlabel('Epoch'); plt.ylabel('Average MAE'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/mae.png')

        plt.figure(3)
        plt.plot(epochs, self.train_feat_l2, label='Train Feature L2')
        if cfg.validation and len(self.val_set) > 0:
            plt.plot(epochs, self.val_feat_l2, label='Val Feature L2')
            plt.legend()
        plt.title('AIRL — Feature L2')
        plt.xlabel('Epoch'); plt.ylabel('Average Feature L2'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/feature_l2.png')

        plt.figure(4)
        plt.plot(epochs, self.train_disc_loss)
        plt.title('AIRL — Discriminator BCE Loss')
        plt.xlabel('Epoch'); plt.ylabel('Disc Loss'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/disc_loss.png')

        plt.figure(5)
        plt.plot(epochs, self.train_expert_acc, label='Expert Acc (D→1)')
        plt.plot(epochs, self.train_policy_acc, label='Policy Acc (D→0)')
        plt.axhline(0.5, color='red', linestyle='--', linewidth=0.8, label='Chance (0.5)')
        plt.title('AIRL — Discriminator Accuracy')
        plt.xlabel('Epoch'); plt.ylabel('Fraction correctly classified'); plt.grid()
        plt.legend()
        plt.savefig(f'./models/{cfg.folder_name}/disc_accuracy.png')

        plt.show()

