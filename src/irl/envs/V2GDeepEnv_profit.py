import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import torch


class V2GDeepEnv(gym.Env):
    """
    Custom Gym Environment for Vehicle-to-Grid (V2G) with Deep MaxEnt IRL support.

    Identical physics to V2GEnv (profit variant) but computes rewards via a 
    neural network R_θ(s, a) instead of a linear weight-feature dot product.
    Feature tracking is retained for monitoring and comparison.

    Reward:
        If reward_net is set:  R = reward_net(obs_normalized, action) - boundary_penalty
        Elif reward_weights:   R = w · φ(s, a) - boundary_penalty  (fallback)
        Else:                  R = 0 - boundary_penalty
    """

    def __init__(self):
        super().__init__()

        # --- REWARD SOURCES ---
        self.reward_net = None          # PyTorch RewardNet (Deep MaxEnt IRL)
        self.initial_states = None

        # --- OBSERVATION NORMALIZATION SCALES ---
        self._obs_scales = np.array([96.0, 1.0, 1.0, 0.47, 2.0, 96.0, 22.0], dtype=np.float32)

        # HIDDEN STATES
        self.day_stage = 0

        # Variable States
        self.timestep = 0
        self.soc = 0
        self.soc_target = 0
        self.location = 0
        self.time_to_next_journey = 0

        # Fixed States per Episode
        self.battery_capacity = 0
        self.home_charge_power = 0
        self.work_charge_power = 0
        self.out_start_timestep = 0
        self.return_start_timestep = 0
        self.journey_distance = 0
        self.out_journey_speed = 0
        self.return_journey_speed = 0

        self.out_duration = 0
        self.return_duration = 0

        self.kwh_per_mile = 1/3
        self.energy_for_out = 0
        self.energy_for_return = 0

        # Helper Maps
        self.battery_capacity_map = {0: 40, 1: 60, 2: 80}
        self.home_charge_power_map = {0: 3, 1: 7.4, 2: 11}
        self.work_charge_power_map = {0: 7.4, 1: 11, 2: 22}

        # Energy price profile
        self.energy_price_profile = np.array([
            0.07, 0.07, 0.07, 0.07, 0.08, 0.08, 0.09, 0.09, 0.10, 0.10, 0.11, 0.12,
            0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.21, 0.22, 0.23, 0.24, 0.26,
            0.27, 0.28, 0.30, 0.31, 0.32, 0.33, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40,
            0.41, 0.42, 0.43, 0.44, 0.44, 0.45, 0.45, 0.46, 0.46, 0.47, 0.47, 0.47,
            0.47, 0.47, 0.47, 0.47, 0.46, 0.46, 0.45, 0.45, 0.44, 0.44, 0.43, 0.42,
            0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.33, 0.32, 0.31, 0.30, 0.28,
            0.27, 0.26, 0.24, 0.23, 0.22, 0.21, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14,
            0.13, 0.12, 0.11, 0.10, 0.10, 0.09, 0.09, 0.08, 0.08, 0.07, 0.07, 0.07
        ])

        self.boundary_penalty_per_kwh = 0.001
        self.action_penalty_coeff = 0.0   # λ for -λ·a² action magnitude penalty
        self.delta_t = 0
        self.action_revenue = 0.0
        self.mean_energy_price = float(np.mean(self.energy_price_profile))
        self.max_energy_price = float(np.max(self.energy_price_profile))
        self.action_price_sum = 0.0
        self.action_price_count = 0

        # --- INFO TRACKING ---
        self.soc_history = []
        self.accumulated_profit = 0.0
        self.soc_change_this_action = 0.0
        self.accumulated_profit_history = []
        self.__reset_feature_history()

        # --- DEFINE SPACES ---
        self.observation_space = spaces.Dict({
            'timestep': spaces.Box(low=0, high=96, shape=(1,), dtype=np.int64),
            'soc': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'soc_target': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'energy_price': spaces.Box(low=0.0, high=0.47, shape=(1,), dtype=np.float32),
            'battery_capacity': spaces.Box(low=0, high=2, shape=(1,), dtype=np.int64),
            'time_to_next_journey': spaces.Box(low=0, high=96, shape=(1,), dtype=np.int64),
            'current_charger_power': spaces.Box(low=0.0, high=22.0, shape=(1,), dtype=np.float32),
        })

        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Reward network helpers                                              #
    # ------------------------------------------------------------------ #

    def _flatten_obs_for_reward(self):
        """Get flattened, normalized observation for the reward network (pre-action state)."""
        if self.location == 0:
            current_charger_power = self.home_charge_power_map[self.home_charge_power]
        else:
            current_charger_power = self.work_charge_power_map[self.work_charge_power]

        raw = np.array([
            self.timestep,
            self.soc,
            self.soc_target,
            self.energy_price_profile[min(self.timestep, 95)],
            self.battery_capacity,
            self.time_to_next_journey,
            current_charger_power,
        ], dtype=np.float32)
        return raw / self._obs_scales

    # ------------------------------------------------------------------ #
    #  Feature tracking (for monitoring, not used in reward computation)   #
    # ------------------------------------------------------------------ #

    def __get_features(self):
        amount_charged = self.soc_change_this_action if self.soc_change_this_action > 0.0 else 0.0
        amount_discharged = -self.soc_change_this_action if self.soc_change_this_action < 0.0 else 0.0

        battery_needed_target = (max(0.0, self.soc_target - self.soc)) ** 2 * self.delta_t
        battery_exceeded_target = (max(0.0, self.soc - self.soc_target)) ** 2 * self.delta_t
        journey_failure = 0.0

        max_price = self.max_energy_price
        if self.action_price_count > 0:
            action_avg_price = self.action_price_sum / self.action_price_count
        else:
            action_avg_price = self.energy_price_profile[min(self.timestep, 95)]

        charge_cost_penalty = 10.0 * amount_charged ** 2 * (action_avg_price / max_price)
        discharge_cost_penalty = 10.0 * amount_discharged ** 2 * ((max_price - action_avg_price) / max_price)

        return np.array([
            amount_charged, amount_discharged,
            charge_cost_penalty, discharge_cost_penalty,
            battery_needed_target, battery_exceeded_target,
            journey_failure,
        ], dtype=np.float32)

    def __reset_feature_history(self):
        self.feature_history = {
            'amount_charged': [], 'amount_discharged': [],
            'charge_cost_penalty': [], 'discharge_cost_penalty': [],
            'battery_needed_target': [], 'battery_exceeded_target': [],
            'journey_failure': [],
        }

    def __update_feature_history(self, features):
        self.feature_history['amount_charged'].append(features[0])
        self.feature_history['amount_discharged'].append(features[1])
        self.feature_history['charge_cost_penalty'].append(features[2])
        self.feature_history['discharge_cost_penalty'].append(features[3])
        self.feature_history['battery_needed_target'].append(features[4])
        self.feature_history['battery_exceeded_target'].append(features[5])
        self.feature_history['journey_failure'].append(features[6])

    def calculate_feature_expectations(self):
        return np.array([
            np.sum(self.feature_history['amount_charged']),
            np.sum(self.feature_history['amount_discharged']),
            np.sum(self.feature_history['charge_cost_penalty']),
            np.sum(self.feature_history['discharge_cost_penalty']),
            np.sum(self.feature_history['battery_needed_target']),
            np.sum(self.feature_history['battery_exceeded_target']),
            np.sum(self.feature_history['journey_failure']),
        ], dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Observation / info helpers                                          #
    # ------------------------------------------------------------------ #

    def __get_obs(self):
        if self.location == 0:
            current_charger_power = self.home_charge_power_map[self.home_charge_power]
        else:
            current_charger_power = self.work_charge_power_map[self.work_charge_power]

        return {
            'timestep': np.array([self.timestep], dtype=np.int64),
            'soc': np.array([self.soc], dtype=np.float32),
            'soc_target': np.array([self.soc_target], dtype=np.float32),
            'energy_price': np.array([self.energy_price_profile[self.timestep]], dtype=np.float32),
            'battery_capacity': np.array([self.battery_capacity], dtype=np.int64),
            'time_to_next_journey': np.array([self.time_to_next_journey], dtype=np.int64),
            'current_charger_power': np.array([current_charger_power], dtype=np.float32),
        }

    def __get_info(self, terminal=False, features=None):
        info = {"delta_t": self.delta_t}
        if terminal:
            info["soc_history"] = self.soc_history
            info["accumulated_profit_history"] = self.accumulated_profit_history
            info["out_start_timestep"] = self.out_start_timestep
            info["return_start_timestep"] = self.return_start_timestep
            info["out_duration"] = self.out_duration
            info["return_duration"] = self.return_duration
            info["features"] = features if features is not None else self.__get_features()
            info["feature_expectation"] = self.calculate_feature_expectations()
        return info

    # ------------------------------------------------------------------ #
    #  Public setters                                                      #
    # ------------------------------------------------------------------ #

    def set_initial_states(self, initial_states) -> None:
        self.initial_states = initial_states

    def set_reward_net(self, reward_net) -> None:
        """Set the neural network reward function."""
        self.reward_net = reward_net

    def set_action_penalty_coeff(self, coeff: float) -> None:
        """Set the action magnitude penalty coefficient λ for -λ·a²."""
        self.action_penalty_coeff = coeff

    # ------------------------------------------------------------------ #
    #  Reset                                                               #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.initial_states is not None:
            self.soc = self.initial_states['soc']
            self.battery_capacity = self.initial_states['battery_capacity']
            self.home_charge_power = self.initial_states['home_charge_power']
            self.work_charge_power = self.initial_states['work_charge_power']
            self.journey_distance = self.initial_states['journey_distance']
            self.out_journey_speed = self.initial_states['out_journey_speed']
            self.return_journey_speed = self.initial_states['return_journey_speed']
            self.out_start_timestep = self.initial_states['out_start_timestep']
            self.return_start_timestep = self.initial_states['return_start_timestep']
            self.out_duration = math.ceil((self.journey_distance / self.out_journey_speed) * 4)
            self.return_duration = math.ceil((self.journey_distance / self.return_journey_speed) * 4)
        else:
            self.soc = max(0.0, self.np_random.normal(loc=0.4944, scale=0.1481))
            self.soc = np.clip(self.soc, 0.0, 1.0)
            self.battery_capacity = self.np_random.choice([0, 1, 2])
            self.home_charge_power = self.np_random.choice([0, 1, 2])
            self.work_charge_power = self.np_random.choice([0, 1, 2])
            self.journey_distance = self.np_random.exponential(scale=10.1331) + 10
            self.out_journey_speed = self.np_random.gamma(8.5518, 3.4713)
            self.return_journey_speed = self.np_random.gamma(8.5518, 3.4713)
            self.journey_distance = np.clip(self.journey_distance, 10, 80)
            self.out_journey_speed = np.clip(self.out_journey_speed, 10, 75)
            self.return_journey_speed = np.clip(self.return_journey_speed, 10, 75)
            self.out_duration = math.ceil((self.journey_distance / self.out_journey_speed) * 4)
            self.return_duration = math.ceil((self.journey_distance / self.return_journey_speed) * 4)
            self.out_start_timestep = math.floor(self.np_random.gamma(11.4837, 2.61428))
            self.out_start_timestep = np.clip(self.out_start_timestep, 0, 80)
            self.return_start_timestep = math.floor(self.np_random.gamma(24.6641, 1.8946) + 20)
            self.return_start_timestep = np.clip(self.return_start_timestep, self.out_start_timestep + self.out_duration + 1, 90)
            self.return_duration = np.clip(self.return_duration, 1, 95 - (self.return_start_timestep + 1))

        self.energy_for_out = (self.journey_distance * self.kwh_per_mile) / self.battery_capacity_map[self.battery_capacity]
        self.energy_for_return = (self.journey_distance * self.kwh_per_mile) / self.battery_capacity_map[self.battery_capacity]

        self.timestep = 0
        self.day_stage = 0
        self.location = 0
        self.soc_target = self.energy_for_out
        self.energy_price = self.energy_price_profile[self.timestep]
        self.time_to_next_journey = self.out_start_timestep

        self.__reset_feature_history()
        self.soc_history = [self.soc]
        self.accumulated_profit = 0.0
        self.accumulated_profit_history = [0.0]

        observation = self.__get_obs()
        info = self.__get_info(terminal=False)
        return observation, info

    # ------------------------------------------------------------------ #
    #  Step                                                                #
    # ------------------------------------------------------------------ #

    def step(self, action):
        # --- Capture pre-action observation for reward network ---
        pre_action_obs_flat = self._flatten_obs_for_reward()

        terminated = False
        truncated = False
        prev_timestep = self.timestep

        self.soc_change_this_action = 0.0
        boundary_violation = 0.0

        self.action_revenue = 0.0
        self.action_price_sum = 0.0
        self.action_price_count = 0

        battery_cap = self.battery_capacity_map[self.battery_capacity]
        if self.location == 0:
            max_charge_power = self.home_charge_power_map[self.home_charge_power]
        else:
            max_charge_power = self.work_charge_power_map[self.work_charge_power]

        action = action[0]
        action = np.round(action, 2)
        action_for_reward = float(action)

        if action == 0.0:
            self.timestep += 1
            self.soc_history.append(self.soc)
            self.accumulated_profit_history.append(self.accumulated_profit)
        else:
            deadline_timestep = None
            match self.day_stage:
                case 0: deadline_timestep = self.out_start_timestep
                case 1: deadline_timestep = self.return_start_timestep
                case 2: deadline_timestep = 95

            charge_amount_remaining = abs(action)
            max_energy_transfer = (max_charge_power * 0.25) / battery_cap
            charging_time = 0

            while self.timestep < deadline_timestep:
                exceeded = False
                energy_transfer = min(max_energy_transfer, charge_amount_remaining)
                charge_amount_remaining -= energy_transfer

                if action > 0:
                    if self.soc + energy_transfer > 0.8:
                        boundary_violation += (self.soc + charge_amount_remaining + energy_transfer - 0.8)
                        energy_transfer = 0.8 - self.soc
                        self.soc = 0.8
                        exceeded = True
                    else:
                        self.soc += energy_transfer
                else:
                    if self.soc <= 0.2:
                        boundary_violation += (energy_transfer + charge_amount_remaining)
                        energy_transfer = 0.0
                        exceeded = True
                    elif self.soc - energy_transfer < 0.2:
                        boundary_violation += (0.2 - (self.soc - charge_amount_remaining - energy_transfer))
                        energy_transfer = self.soc - 0.2
                        self.soc = 0.2
                        exceeded = True
                    else:
                        self.soc -= energy_transfer

                self.timestep += 1
                charging_time += 1
                self.soc_history.append(self.soc)
                self.soc_change_this_action += energy_transfer if action > 0 else -energy_transfer

                energy_price = self.energy_price_profile[self.timestep - 1]
                if energy_transfer > 0:
                    self.action_price_sum += energy_price
                    self.action_price_count += 1

                self.action_revenue += energy_transfer * battery_cap * energy_price if action < 0 else -energy_transfer * battery_cap * energy_price
                if action > 0:
                    self.accumulated_profit -= energy_transfer * battery_cap * energy_price
                else:
                    self.accumulated_profit += energy_transfer * battery_cap * energy_price

                self.accumulated_profit_history.append(self.accumulated_profit)
                if exceeded or charge_amount_remaining <= 0.0:
                    break

        self.delta_t = self.timestep - prev_timestep

        # Features (for monitoring only)
        features = self.__get_features()

        # --- UPDATE LOCATION AND DAY STAGE ---
        if self.day_stage == 0 and self.timestep >= self.out_start_timestep:
            self.location = 1
            self.day_stage = 1
            self.timestep = self.out_start_timestep + self.out_duration
            self.time_to_next_journey = max(0, self.return_start_timestep - self.timestep)
            self.soc_target = self.energy_for_return

            energy_used = min(self.energy_for_out, self.soc)
            self.soc = max(0.0, self.soc - energy_used)
            if energy_used < self.energy_for_out:
                features[6] = 1.0

            for _ in range(self.out_duration):
                self.soc_history.append(self.soc_history[-1] - (energy_used / self.out_duration))

        elif self.day_stage == 1 and self.timestep >= self.return_start_timestep:
            self.location = 0
            self.day_stage = 2
            self.timestep = max(0, self.return_start_timestep + self.return_duration)
            self.time_to_next_journey = 96 - self.timestep
            self.soc_target = self.energy_for_out

            energy_used = min(self.energy_for_return, self.soc)
            self.soc = max(0.0, self.soc - energy_used)
            if energy_used < self.energy_for_return:
                features[6] = 1.0

            for _ in range(self.return_duration):
                self.soc_history.append(self.soc_history[-1] - (energy_used / self.return_duration))

        elif self.day_stage == 2 and self.timestep >= 95:
            self.timestep = 95
            terminated = True

        else:
            self.time_to_next_journey = max(0, self.time_to_next_journey - (charging_time if action != 0.0 else 1))

        self.__update_feature_history(features)

        # --- REWARD ---
        reward = 0.0

        if self.reward_net is not None:
            with torch.no_grad():
                obs_t = torch.tensor(pre_action_obs_flat, dtype=torch.float32).unsqueeze(0)
                act_t = torch.tensor([[action_for_reward]], dtype=torch.float32)
                # Scale by delta_t so that total episode reward is proportional to the
                # time-integral of R(s,a), not the number of variable-length actions taken.
                # Without this, PPO learns to take many tiny actions to accumulate more
                # reward events rather than large, expert-like charge/discharge actions.
                reward = self.reward_net(obs_t, act_t).item() * self.delta_t

        if boundary_violation > 0.0:
            reward -= self.boundary_penalty_per_kwh * boundary_violation * battery_cap * self.delta_t

        # Action magnitude penalty: -λ·a²  (also scaled by delta_t for consistency)
        if self.action_penalty_coeff > 0.0:
            reward -= self.action_penalty_coeff * (action_for_reward ** 2) * self.delta_t

        observation = self.__get_obs()
        info = self.__get_info(terminal=terminated, features=features)

        return observation, reward, terminated, truncated, info
