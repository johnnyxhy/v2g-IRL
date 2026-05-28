"""
Adversarial Inverse Reinforcement Learning for V2G continuous environment.

Based on: Fu, Luo, Levine (2018)
    "Learning Robust Rewards with Adversarial Inverse Reinforcement Learning"
    https://arxiv.org/abs/1710.11248
"""

import numpy as np
import json
import csv
import gymnasium as gym
from dataclasses import dataclass
from sbx import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from irl.Adversarial.Adversarial import OBS_SCALES, FlattenNormalizeObsWrapper, BaseAdversarialTrainer
from irl.utils.tools import compute_dtw, compute_mae
from tqdm import tqdm
import os

import torch
import torch.nn.functional as F
import jax.numpy as jnp


# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #

class AdversarialConfig:
    """Configuration for Adversarial IRL training with SAC inner loop (continuous environment)."""

    # IRL outer loop
    n_epochs: int = 10
    disc_lr: float = 1e-3               # discriminator (g_θ + h_φ) learning rate
    disc_lr_end: float = 1e-4           # final LR after cosine annealing
    rollout_samples: int = 10           # policy rollouts per expert trajectory for disc

    # Discriminator network dimensions
    reward_hidden_dim: int = 64
    shaping_hidden_dim: int = 64
    reward_obs_dim: int = 7
    reward_action_dim: int = 1

    # Warm-start from a previous run
    pretrained_reward_net_path: str = None
    pretrained_shaping_net_path: str = None

    # SAC inner loop
    policy_train_steps_per_iter: int = 100_000
    policy_train_lr: float = 3e-4
    policy_gamma: float = 1.0
    policy_batch_size: int = 64
    policy_buffer_size: int = 100_000
    policy_learning_starts: int = 1_000
    policy_ent_coef: str = 'auto'

    # Data
    train_ratio: float = 0.8
    segment: str = None

    # Discriminator training epochs per IRL epoch.
    disc_epochs: int = 1

    # Discriminator regularisation
    disc_l2_reg: float = 0.01

    # Action magnitude penalty applied inside the env (does not affect disc gradient).
    action_penalty_coeff: float = 0.0

    # Reward scale applied to f inside the env.  Scales the PPO/SAC reward signal
    # without changing the discriminator's logit or gradient.
    reward_scale: float = 1.0

    # Behavioral Cloning pre-training.
    bc_pretrain_steps: int = 0   # number of Adam gradient steps (0 = disabled)
    bc_lr: float = 1e-3

    # Number of parallel envs for SAC training.
    n_envs: int = 1

    # Run validation rollouts on the held-out set each epoch.
    validation: bool = False

    # Human-readable description written to about.md in the model folder.
    description: str = ""

    # Saving
    folder_name: str = "Adversarial_continuous"


# ------------------------------------------------------------------ #
#  Expert data loading                                                 #
# ------------------------------------------------------------------ #

@dataclass
class AdversarialExpertTrajectory:
    """
    Expert demonstration in Adversarial IRL format for the continuous environment.
    """
    episodeID: int
    segment: str
    initial_values: dict
    soc_history: list
    observations: np.ndarray        # (N, obs_dim) raw, unnormalized  s
    next_observations: np.ndarray   # (N, obs_dim) raw, unnormalized  s'
    actions: np.ndarray             # (N, 1) continuous float in [-1, 1]
    delta_ts: np.ndarray            # (N,)  number of env timesteps per action
    dones: np.ndarray               # (N,)  float: 1.0 only for terminal action
    feature_expectation: np.ndarray # (n_features,) for monitoring only


def load_adversarial_expert_data(json_path: str, segment=None, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    Load Adversarial IRL-format trajectories from a JSON file produced by
    expert_loader_airl_continuous.py.

    Returns:
        train_set, val_set, test_set — lists of AdversarialExpertTrajectory
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    trajectories = []
    for traj in data:
        sap = traj['state_action_pairs']
        expert = AdversarialExpertTrajectory(
            episodeID=traj['episodeID'],
            segment=traj['segment'],
            initial_values=traj['initial_values'],
            soc_history=traj['soc_history'],
            observations=np.array(sap['observations'],      dtype=np.float32),
            next_observations=np.array(sap['next_observations'], dtype=np.float32),
            actions=np.array(sap['actions'],                dtype=np.float32),
            delta_ts=np.array(sap['delta_ts'],              dtype=np.float32),
            dones=np.array(sap['dones'],                    dtype=np.float32),
            feature_expectation=np.array(traj['feature_expectation'], dtype=np.float32),
        )
        trajectories.append(expert)

    if segment is not None:
        trajectories = [t for t in trajectories if segment in t.segment]

    n_train = int(len(trajectories) * train_ratio)
    n_val = int(len(trajectories) * val_ratio)
    return trajectories[:n_train], trajectories[n_train:n_train + n_val], trajectories[n_train + n_val:]


# ------------------------------------------------------------------ #
#  Adversarial IRL Trainer (continuous)                                #
# ------------------------------------------------------------------ #

class AdversarialTrainer(BaseAdversarialTrainer):
    """
    Adversarial IRL trainer for V2GDeepEnv (profit) using SBX SAC.
    """

    plot_title_prefix = "Adversarial IRL (continuous)"

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

        rollout_base_env = gym.make(self.env_name)
        rollout_base_env = FlattenNormalizeObsWrapper(rollout_base_env)
        rollout_env = DummyVecEnv([lambda: rollout_base_env])

        def _make_sac(epoch_idx: int):
            return SAC(
                policy="MlpPolicy",
                env=vec_env,
                verbose=0,
                learning_rate=cfg.policy_train_lr,
                gamma=cfg.policy_gamma,
                batch_size=cfg.policy_batch_size,
                buffer_size=cfg.policy_buffer_size,
                learning_starts=cfg.policy_learning_starts,
                ent_coef=cfg.policy_ent_coef,
                seed=42 + epoch_idx,
                train_freq=1,
                gradient_steps=1,
                tensorboard_log=f"./models/{cfg.folder_name}/tensorboard/",
            )

        model = _make_sac(0)

        # Optional BC pre-training
        if cfg.bc_pretrain_steps > 0:
            self._bc_pretrain(model)

        for epoch in range(cfg.n_epochs):
            print(f"\nEpoch {epoch+1}/{cfg.n_epochs}")

            # ---- 1. SAC training under current AIRL reward ----
            # Push g_θ and h_φ into all envs.
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

            # Reset SAC entropy coefficient each epoch so auto-tuning restarts fresh.
            if hasattr(model, 'ent_coef_state'):
                model.ent_coef_state = model.ent_coef_state.replace(
                    params={'log_ent_coef': jnp.array(-2.3026)}  # ≈ 0.1
                )

            model.learn(
                total_timesteps=cfg.policy_train_steps_per_iter,
                tb_log_name=f"epoch_{epoch+1}",
                progress_bar=True,
                reset_num_timesteps=False,
                log_interval=500,
            )

            # ---- 2. Discriminator update ----
            N = len(self.train_set)
            obs_scales = self.obs_scales

            avg_dtw     = 0.0
            avg_mae     = 0.0
            avg_feat_l2 = 0.0

            collected = []
            for traj in tqdm(self.train_set, desc="Collecting rollouts"):
                expert_obs      = torch.tensor(traj.observations,      dtype=torch.float32) / obs_scales
                expert_next_obs = torch.tensor(traj.next_observations, dtype=torch.float32) / obs_scales
                expert_act      = torch.tensor(traj.actions,           dtype=torch.float32)  # (T,1)
                expert_dt       = torch.tensor(traj.delta_ts,          dtype=torch.float32)  # (T,)
                expert_dones    = torch.tensor(traj.dones,             dtype=torch.float32)  # (T,)

                log_pi_expert = torch.tensor(
                    self._get_log_prob(model, expert_obs.numpy(), traj.actions),
                    dtype=torch.float32,
                )  # (T,)

                policy_obs_all      = []
                policy_next_obs_all = []
                policy_act_all      = []
                policy_dt_all       = []
                policy_done_all     = []

                for _ in range(cfg.rollout_samples):
                    (obs_list, next_obs_list, act_list, dones_list,
                     dt_list, _, soc_hist, feat_exp) = self._do_rollout(rollout_env, model, traj)

                    policy_obs_all.extend(obs_list)
                    policy_next_obs_all.extend(next_obs_list)
                    policy_act_all.extend(act_list)
                    policy_done_all.extend(dones_list)
                    policy_dt_all.extend(dt_list)

                    expert_soc = np.array(traj.soc_history, dtype=np.float32)
                    avg_dtw += compute_dtw(expert_soc, np.array(soc_hist, dtype=np.float32)) / (
                        N * cfg.rollout_samples
                    )
                    avg_mae += compute_mae(expert_soc, np.array(soc_hist, dtype=np.float32)) / (
                        N * cfg.rollout_samples
                    )
                    avg_feat_l2 += np.linalg.norm(traj.feature_expectation - feat_exp) / (
                        N * cfg.rollout_samples
                    )

                policy_obs_t      = torch.tensor(np.array(policy_obs_all),      dtype=torch.float32)
                policy_next_obs_t = torch.tensor(np.array(policy_next_obs_all), dtype=torch.float32)
                policy_act_t      = torch.tensor(np.array(policy_act_all),      dtype=torch.float32)
                policy_dt_t       = torch.tensor(np.array(policy_dt_all),       dtype=torch.float32)
                policy_done_t     = torch.tensor(np.array(policy_done_all),     dtype=torch.float32)

                log_pi_policy = torch.tensor(
                    self._get_log_prob(
                        model,
                        policy_obs_t.numpy(),
                        np.array(policy_act_all, dtype=np.float32),
                    ),
                    dtype=torch.float32,
                )  # (B,)

                collected.append((
                    expert_obs, expert_next_obs, expert_act, expert_dt, expert_dones, log_pi_expert,
                    policy_obs_t, policy_next_obs_t, policy_act_t, policy_dt_t, policy_done_t, log_pi_policy,
                ))

            # Phase B — train discriminator
            avg_disc_loss  = 0.0
            avg_expert_acc = 0.0
            avg_policy_acc = 0.0

            diag_f_expert_mean  = 0.0
            diag_f_policy_mean  = 0.0
            diag_g_expert_mean  = 0.0
            diag_g_policy_mean  = 0.0
            diag_h_expert_mean  = 0.0
            diag_h_policy_mean  = 0.0
            diag_logpi_e_mean   = 0.0
            diag_logpi_p_mean   = 0.0
            diag_logit_e_mean   = 0.0
            diag_logit_p_mean   = 0.0
            total_grad_norm     = 0.0

            for disc_ep in tqdm(range(cfg.disc_epochs), desc="Disc epochs"):
                ep_loss       = 0.0
                ep_expert_acc = 0.0
                ep_policy_acc = 0.0
                is_last = (disc_ep == cfg.disc_epochs - 1)

                for (expert_obs, expert_next_obs, expert_act,
                     expert_dt, expert_dones, log_pi_expert,
                     policy_obs_t, policy_next_obs_t, policy_act_t,
                     policy_dt_t, policy_done_t, log_pi_policy) in collected:

                    self.disc_optimizer.zero_grad()

                    f_expert = self._compute_f(
                        expert_obs, expert_act,
                        expert_next_obs, expert_dones, expert_dt,
                    )  # (T,)
                    f_policy = self._compute_f(
                        policy_obs_t, policy_act_t,
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
                    loss_traj = (loss_expert + loss_policy) / 2
                    loss_traj.backward()

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

                    if is_last:
                        with torch.no_grad():
                            diag_f_expert_mean += f_expert.detach().mean().item() / N
                            diag_f_policy_mean += f_policy.detach().mean().item() / N
                            diag_g_expert_mean += self.reward_net(expert_obs, expert_act).mean().item() / N
                            diag_g_policy_mean += self.reward_net(policy_obs_t, policy_act_t).mean().item() / N
                            diag_h_expert_mean += self.shaping_net(expert_obs).mean().item() / N
                            diag_h_policy_mean += self.shaping_net(policy_obs_t).mean().item() / N
                            diag_logpi_e_mean  += log_pi_expert.mean().item() / N
                            diag_logpi_p_mean  += log_pi_policy.mean().item() / N
                            diag_logit_e_mean  += logit_expert.detach().mean().item() / N
                            diag_logit_p_mean  += logit_policy.detach().mean().item() / N

                avg_disc_loss  += ep_loss        / cfg.disc_epochs
                avg_expert_acc += ep_expert_acc  / cfg.disc_epochs
                avg_policy_acc += ep_policy_acc  / cfg.disc_epochs

            self.disc_scheduler.step()
            current_lr = self.disc_scheduler.get_last_lr()[0]

            # ---- 3. Validation ----
            val_dtw = 0.0
            val_mae = 0.0
            val_feat_l2 = 0.0
            if cfg.validation and len(self.val_set) > 0:
                M = len(self.val_set)
                for traj in tqdm(self.val_set, desc="Validation"):
                    (_, _, _, _, _, _, soc_hist, feat_exp) = self._do_rollout(
                        rollout_env, model, traj, deterministic=False
                    )
                    val_dtw += compute_dtw(
                        np.array(traj.soc_history, dtype=np.float32),
                        np.array(soc_hist, dtype=np.float32),
                    ) / M
                    val_mae += compute_mae(
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

            # ---- 5. Save metrics ----
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

            model.save(f"./models/{cfg.folder_name}/sac_epoch{epoch+1}")
            torch.save(
                self.reward_net.state_dict(),
                f"./models/{cfg.folder_name}/reward_net_epoch{epoch+1}.pt",
            )
            torch.save(
                self.shaping_net.state_dict(),
                f"./models/{cfg.folder_name}/shaping_net_epoch{epoch+1}.pt",
            )

        self._plot_results()

        if self.test_set:
            self._evaluate_test_set(rollout_env, model, n_rollouts=30)

        print("Training completed.")

    # -------------------------------------------------------------- #
    #  Policy helpers                                                  #
    # -------------------------------------------------------------- #

    def _bc_pretrain(self, model: SAC) -> None:
        """
        Behavioural Cloning pre-training for the SBX SAC actor via optax.

        Maximises E[log π(a_expert | s_expert)] on the training set, placing the
        policy near the expert's action distribution before Adversarial IRL starts.

        For continuous actions, a_expert are squashed floats in [-1, 1].  The SAC
        actor returns a distrax distribution (squashed Gaussian) and log_prob
        accounts for the tanh Jacobian correction automatically.
        """
        import jax
        import optax

        cfg = self.cfg
        obs_scale = np.array(OBS_SCALES, dtype=np.float32)

        all_obs = np.concatenate([
            t.observations / obs_scale for t in self.train_set
        ])  # (N_total, obs_dim)
        all_acts = np.concatenate([
            t.actions for t in self.train_set
        ], axis=0).astype(np.float32)  # (N_total, 1) squashed actions in [-1, 1]

        apply_fn  = model.policy.actor_state.apply_fn
        bc_optim  = optax.adam(cfg.bc_lr)
        bc_opt_st = bc_optim.init(model.policy.actor_state.params)

        @jax.jit
        def _step(params, opt_state, obs_b, act_b):
            def loss_fn(p):
                dist = apply_fn(p, obs_b)
                # act_b: (B, 1) — TFP distributions require (batch, action_dim)
                lp = dist.log_prob(act_b)
                return -lp.mean()
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

        # Reset actor optimizer state so BC momentum doesn't bias first epoch.
        try:
            fresh_opt_state = model.policy.actor_state.tx.init(
                model.policy.actor_state.params
            )
            model.policy.actor_state = model.policy.actor_state.replace(
                opt_state=fresh_opt_state
            )
        except AttributeError:
            pass

        print("BC pre-training done.")

    def _get_log_prob(
        self,
        model: SAC,
        obs_np: np.ndarray,    # (B, obs_dim) float32, normalized
        acts_np: np.ndarray,   # (B, 1) or (B,) float32, squashed actions in [-1, 1]
    ) -> np.ndarray:
        """
        Evaluate log π(a|s) using the current SAC actor (squashed Gaussian).

        SBX SAC uses JAX/Flax.  actor_state.apply_fn returns a distrax distribution
        (e.g. Transformed(Normal, Tanh)).  log_prob correctly accounts for the tanh
        Jacobian change-of-variables so acts_np should be squashed (in [-1, 1]).
        """
        obs_jax = jnp.array(obs_np)
        # TFP distributions require (batch, action_dim) — keep the trailing 1
        act_jax = jnp.array(acts_np.reshape(-1, 1), dtype=jnp.float32)  # (B, 1)
        dist = model.policy.actor_state.apply_fn(
            model.policy.actor_state.params, obs_jax
        )
        lp = dist.log_prob(act_jax)
        return np.array(lp, dtype=np.float32)

    # -------------------------------------------------------------- #
    #  Rollout helper                                                  #
    # -------------------------------------------------------------- #

    def _do_rollout(self, rollout_env, model, traj, deterministic=False):
        """
        Run one episode from the expert's initial state.

        Returns:
            obs_list, next_obs_list, act_list, dones_list,
            dt_list, traj_reward, soc_history, feature_expectation
        """
        rollout_env.envs[0].unwrapped.set_initial_states(traj.initial_values)
        obs = rollout_env.reset()

        obs_list      = []
        next_obs_list = []
        act_list      = []
        dones_list    = []
        dt_list       = []
        traj_reward   = 0.0

        while True:
            obs_flat = obs[0].copy()  # (obs_dim,) normalized — this is s
            action, _ = model.predict(obs, deterministic=deterministic)
            # action shape: (1, 1) from SAC with 1-D Box action space
            act_float = float(action.reshape(-1)[0])  # scalar in [-1, 1]

            next_obs, reward, dones_arr, infos = rollout_env.step(action)
            done = bool(dones_arr[0])

            obs_list.append(obs_flat)
            next_obs_list.append(next_obs[0].copy())  # s' (normalized by wrapper)
            act_list.append([act_float])               # (1,) for reward net input
            dones_list.append(float(done))
            dt_list.append(infos[0]['delta_t'])
            traj_reward += float(reward[0])

            obs = next_obs
            if done:
                break

        soc_history = infos[0]['soc_history']
        feature_expectation = np.array(infos[0]['feature_expectation'], dtype=np.float32)

        return (
            obs_list, next_obs_list, act_list, dones_list,
            dt_list, traj_reward, soc_history, feature_expectation,
        )
