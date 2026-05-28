import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class V2GEnv(gym.Env):
    """
    Custom Gym Environment for Vehicle-to-Grid (V2G)

    Note:
    - Simple setup with discrete action space (charge / discharge / idle)
    - Charging restricted to SoC range [0.2, 0.8]

    Feature Space:
    - amount_charged: SoC increase this action (always >= 0)
    - amount_discharged: SoC decrease this action (always >= 0)
    - battery_needed_target: squared deficit from target SoC x delta_t
    - battery_exceeded_target: squared surplus above target SoC x delta_t
    - journey_failure: 1 if unable to complete journey, else 0

    """
    def __init__(self):

        super().__init__()

        # --- DEFINE INTERNAL STATES ---
        self.initial_states = None
        self.reward_weights = None

        # HIDDEN STATES
        self.day_stage = 0                  # 0: before work, 1: work, 2: after work

        # Variable States
        self.timestep = 0                   # 0 - 95
        self.soc = 0                        # 0.0 - 1.0
        self.soc_target = 0                 # 0.0 - 1.0
        self.soc_gap = 0                    # soc_target - soc
        self.location = 0                   # 0: home, 1: work
        self.time_to_next_journey = 0       # 0 - 95

        # Fixed States per Episode
        self.battery_capacity = 0           # 0: 40 kWh, 1: 60 kWh, 2: 80 kWh
        self.home_charge_power = 0          # 0: 3 kW, 1: 7.4 kW, 2: 11 kW
        self.work_charge_power = 0          # 0: 7.4 kW, 1: 11 kW, 2: 22 kW
        self.out_start_timestep = 0         # 0 - 95
        self.return_start_timestep = 0      # 0 - 95
        self.journey_distance = 0           # in miles
        self.out_journey_speed = 0          # in mph
        self.return_journey_speed = 0       # in mph

        self.out_duration = 0               # 0 - 95
        self.return_duration = 0            # 0 - 95

        self.kwh_per_mile = 1/3             # Assume 3 miles per kWh

        self.energy_for_out = 0             # SoC needed for outgoing journey
        self.energy_for_return = 0          # SoC needed for return journey

        # Helper Maps
        self.battery_capacity_map = {0: 40, 1: 60, 2: 80}  # kWh
        self.home_charge_power_map = {0: 3, 1: 7.4, 2: 11}  # kW
        self.work_charge_power_map = {0: 7.4, 1: 11, 2: 22} # kW

        # ASSUME FIXED ENERGY PRICE PROFILE THROUGHOUT DAY
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

        # --- Boundary penalty ---
        self.boundary_penalty_per_kwh = 0.00

        # --- Reward gamma handling ---
        self.delta_t = 0

        # --- SoC change tracking per action ---
        self.soc_change_this_action = 0.0

        # --- INFO TRACKING ---
        self.soc_history = []
        self.accumulated_profit = 0.0
        self.accumulated_profit_history = []
        self.__reset_feature_history()

        # --- DEFINE SPACES ---

        self.observation_space = spaces.Dict({
            'timestep': spaces.Box(low=0, high=96, shape=(1,), dtype=np.int64),
            'soc': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'soc_gap': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'energy_price': spaces.Box(low=0.0, high=0.47, shape=(1,), dtype=np.float32),
            'battery_capacity': spaces.Box(low=0, high=2, shape=(1,), dtype=np.int64),
            'time_to_next_journey': spaces.Box(low=0, high=96, shape=(1,), dtype=np.int64),
            'current_charger_power': spaces.Box(low=0.0, high=22.0, shape=(1,), dtype=np.float32),
        })

        self.action_space = spaces.Discrete(3)  # 0: charge, 1: discharge, 2: idle


    def __get_features(self):
        """
        Get current feature values as a numpy array

        Returns:
            features (np.array): Array of feature values [5]
        """

        # 1-2. amount_charged and amount_discharged
        amount_charged = self.soc_change_this_action if self.soc_change_this_action > 0.0 else 0.0
        amount_discharged = -self.soc_change_this_action if self.soc_change_this_action < 0.0 else 0.0

        # 3. battery_needed_target: squared deficit x delta_t
        battery_needed_target = (max(0.0, self.soc_target - self.soc)) ** 2 * self.delta_t

        # 4. battery_exceeded_target: squared surplus x delta_t
        battery_exceeded_target = (max(0.0, self.soc - self.soc_target)) ** 2 * self.delta_t

        # 5. journey_failure: set in step() after journey transitions
        journey_failure = 0.0

        return np.array([
            amount_charged,
            amount_discharged,
            battery_needed_target,
            battery_exceeded_target,
            journey_failure,
        ], dtype=np.float32)

    def __reset_feature_history(self):
        self.feature_history = {
            'amount_charged': [],
            'amount_discharged': [],
            'battery_needed_target': [],
            'battery_exceeded_target': [],
            'journey_failure': [],
        }

    def __update_feature_history(self, features):
        self.feature_history['amount_charged'].append(features[0])
        self.feature_history['amount_discharged'].append(features[1])
        self.feature_history['battery_needed_target'].append(features[2])
        self.feature_history['battery_exceeded_target'].append(features[3])
        self.feature_history['journey_failure'].append(features[4])

    def calculate_feature_expectations(self):
        """
        Calculate feature expectations over the episode

        Returns:
            feature_expectations (np.array): Array of feature expectations [5]
        """

        return np.array([
            np.sum(self.feature_history['amount_charged']),
            np.sum(self.feature_history['amount_discharged']),
            np.sum(self.feature_history['battery_needed_target']),
            np.sum(self.feature_history['battery_exceeded_target']),
            np.sum(self.feature_history['journey_failure']),
        ], dtype=np.float32)

    def __get_obs(self):
        """
        Get observations as a dictionary
        """

        if self.location == 0:  # At home
            current_charger_power = self.home_charge_power_map[self.home_charge_power]
        else:                   # At work
            current_charger_power = self.work_charge_power_map[self.work_charge_power]

        return {
            'timestep': np.array([self.timestep], dtype=np.int64),
            'soc': np.array([self.soc], dtype=np.float32),
            'soc_gap': np.array([self.soc_gap], dtype=np.float32),
            'energy_price': np.array([self.energy_price_profile[self.timestep]], dtype=np.float32),
            'battery_capacity': np.array([self.battery_capacity], dtype=np.int64),
            'time_to_next_journey': np.array([self.time_to_next_journey], dtype=np.int64),
            'current_charger_power': np.array([current_charger_power], dtype=np.float32),
        }

    def __get_info(self, terminal=False, features=None):
        """
        Get additional info as a dictionary.
        Expensive fields only included at terminal steps.
        """
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

    def set_initial_states(self, initial_states):
        """
        Set initial states for the environment

        Args:
            initial_states (dict): Dictionary of initial states to set
        """
        self.initial_states = initial_states

    def set_reward_weights(self, reward_weights):
        """
        Set reward weights for the environment

        Args:
            reward_weights (np.array): Array of reward weights [5]
        """
        self.reward_weights = reward_weights

    def reset(self, seed=None, options=None):
        """
        Start a new episode

        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional information for the environment.

        Returns:
            observation (dict): Initial observation of the environment.
            info (dict): Additional information.
        """
        super().reset(seed=seed)

        # --- INITIALISE REWARD WEIGHTS ---
        if self.reward_weights is None:
            self.reward_weights = np.array([0.5, 0.5, -1.0, -1.0, -1.0], dtype=np.float32)

        # --- INITIALISE STATES ---

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
            self.soc = np.clip(self.np_random.normal(loc=0.4944, scale=0.1481), 0.0, 1.0)

            self.battery_capacity = self.np_random.choice([0, 1, 2])
            self.home_charge_power = self.np_random.choice([0, 1, 2])
            self.work_charge_power = self.np_random.choice([0, 1, 2])

            self.journey_distance = np.clip(self.np_random.exponential(scale=10.1331) + 10, 10, 80)
            self.out_journey_speed = np.clip(self.np_random.gamma(8.5518, 3.4713), 10, 75)
            self.return_journey_speed = np.clip(self.np_random.gamma(8.5518, 3.4713), 10, 75)

            self.out_duration = math.ceil((self.journey_distance / self.out_journey_speed) * 4)
            self.return_duration = math.ceil((self.journey_distance / self.return_journey_speed) * 4)

            self.out_start_timestep = int(np.clip(math.floor(self.np_random.gamma(11.4837, 2.61428)), 0, 80))
            self.return_start_timestep = int(np.clip(
                math.floor(self.np_random.gamma(24.6641, 1.8946) + 20),
                self.out_start_timestep + self.out_duration + 1, 90
            ))
            self.return_duration = int(np.clip(self.return_duration, 1, 95 - (self.return_start_timestep + 1)))

        # --- ADDITIONAL CALCULATIONS ---
        self.energy_for_out = (self.journey_distance * self.kwh_per_mile) / self.battery_capacity_map[self.battery_capacity]
        self.energy_for_return = (self.journey_distance * self.kwh_per_mile) / self.battery_capacity_map[self.battery_capacity]

        self.timestep = 0
        self.day_stage = 0
        self.location = 0
        self.soc_target = self.energy_for_out + 0.2
        self.soc_gap = self.soc_target - self.soc
        self.time_to_next_journey = self.out_start_timestep
        self.delta_t = 0
        self.soc_change_this_action = 0.0

        # --- CLEAR FEATURE HISTORY ---
        self.__reset_feature_history()

        # --- RESET TRACKING INFO ---
        self.soc_history = [self.soc]
        self.accumulated_profit = 0.0
        self.accumulated_profit_history = [0.0]

        observation = self.__get_obs()
        info = self.__get_info(terminal=False)

        return observation, info

    def step(self, action):
        """
        Take an action in the environment

        Args:
            action (int): 0 = charge, 1 = discharge, 2 = idle

        Returns:
            observation (dict): Next observation of the environment.
            reward (float): Reward obtained from the action.
            done (bool): Whether the episode has ended.
            info (dict): Additional information.
        """

        terminated = False
        truncated = False
        prev_timestep = self.timestep

        self.soc_change_this_action = 0.0
        boundary_violation = 0.0

        battery_cap = self.battery_capacity_map[self.battery_capacity]
        if self.location == 0:  # At home
            max_charge_power = self.home_charge_power_map[self.home_charge_power]
        else:                   # At work
            max_charge_power = self.work_charge_power_map[self.work_charge_power]

        # --- UPDATE TIMESTEP AND SOC ---

        if action == 2:  # Idle
            self.timestep += 1
            self.soc_history.append(self.soc)
            self.accumulated_profit_history.append(self.accumulated_profit)

        else:
            deadline_timestep = None
            match self.day_stage:
                case 0:  # Before work
                    deadline_timestep = self.out_start_timestep
                case 1:  # At work
                    deadline_timestep = self.return_start_timestep
                case 2:  # After work
                    deadline_timestep = 95

            for _ in range(5):  # Run for maximum of 5 timesteps
                if self.timestep >= deadline_timestep:
                    break

                if action == 0:  # Charge
                    if self.soc >= 0.8:
                        boundary_violation += (max_charge_power * 0.25) / battery_cap
                        self.timestep += 1
                        self.soc_history.append(self.soc)
                        self.accumulated_profit_history.append(self.accumulated_profit)
                        break

                    max_energy_added = (max_charge_power * 0.25) / battery_cap
                    energy_added = min(max_energy_added, 0.8 - self.soc)
                    self.soc += energy_added
                    self.soc = np.clip(self.soc, 0.0, 1.0)
                    self.soc_change_this_action += energy_added
                    self.accumulated_profit -= energy_added * battery_cap * self.energy_price_profile[self.timestep]

                elif action == 1:  # Discharge
                    if self.soc <= 0.2:
                        boundary_violation += (max_charge_power * 0.25) / battery_cap
                        self.timestep += 1
                        self.soc_history.append(self.soc)
                        self.accumulated_profit_history.append(self.accumulated_profit)
                        break

                    max_energy_removed = (max_charge_power * 0.25) / battery_cap
                    energy_removed = min(max_energy_removed, self.soc - 0.2)
                    self.soc -= energy_removed
                    self.soc = np.clip(self.soc, 0.0, 1.0)
                    self.soc_change_this_action -= energy_removed
                    self.accumulated_profit += energy_removed * battery_cap * self.energy_price_profile[self.timestep]

                self.soc_history.append(self.soc)
                self.accumulated_profit_history.append(self.accumulated_profit)
                self.timestep += 1

        # --- TRACK TIME DISCOUNTING ---
        self.delta_t = self.timestep - prev_timestep

        # --- GET FEATURES ---
        features = self.__get_features()

        # --- UPDATE LOCATION AND DAY STAGE ---
        if self.day_stage == 0 and self.timestep >= self.out_start_timestep:

            self.location = 1  # Move to work
            self.day_stage = 1
            self.timestep = self.out_start_timestep + self.out_duration
            self.time_to_next_journey = max(0, self.return_start_timestep - self.timestep)
            self.soc_target = self.energy_for_return + 0.2

            energy_used = min(self.energy_for_out, self.soc)
            self.soc = max(0.0, self.soc - energy_used)

            if energy_used < self.energy_for_out:
                features[4] = 1.0  # journey_failure

            for _ in range(self.out_duration):
                self.soc_history.append(self.soc_history[-1] - (energy_used / self.out_duration))

        elif self.day_stage == 1 and self.timestep >= self.return_start_timestep:

            self.location = 0  # Move to home
            self.day_stage = 2
            self.timestep = self.return_start_timestep + self.return_duration
            self.time_to_next_journey = 96 - self.timestep
            self.soc_target = self.energy_for_out + 0.2

            energy_used = min(self.energy_for_return, self.soc)
            self.soc = max(0.0, self.soc - energy_used)

            if energy_used < self.energy_for_return:
                features[4] = 1.0  # journey_failure

            for _ in range(self.return_duration):
                self.soc_history.append(self.soc_history[-1] - (energy_used / self.return_duration))

        elif self.day_stage == 2 and self.timestep >= 95:
            self.timestep = 95
            terminated = True

        else:
            self.time_to_next_journey = max(0, self.time_to_next_journey - self.delta_t)

        self.soc_gap = self.soc_target - self.soc

        # --- UPDATE FEATURE HISTORY ---
        self.__update_feature_history(features)

        # --- REWARD ---
        reward = float(np.dot(self.reward_weights, features))

        if boundary_violation > 0.0:
            reward -= self.boundary_penalty_per_kwh * boundary_violation * battery_cap

        # --- GET NEXT OBSERVATION ---
        observation = self.__get_obs()
        info = self.__get_info(terminal=terminated, features=features)

        if not self.observation_space.contains(observation):
            print("INVALID OBSERVATION!")
            print("obs:", observation)
            raise ValueError("Observation not in observation_space")

        return observation, reward, terminated, truncated, info
