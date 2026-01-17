import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class V2GEnv(gym.Env):
    """
    Custom Gym Environment for Vehicle-to-Grid (V2G) 

    Note:
    - Continuous observation spaces for SoC, SoC_target, energy price
    - Continuous action space: Charge/discharge by percentage

    """
    def __init__(self):

        # --- DEFINE INTERNAL STATES ---
        # Values will be initialised in reset()

        # HIDDEN STATES
        self.day_stage = 0                  # 0: before work, 1: work, 2: after work

        # Variable States
        self.timestep = 0                   # 0 - 95
        self.soc = 0                        # 0.0 - 1.0
        self.soc_target = 0                 # 0.0 - 1.0
        self.location = 0                   # 0: home, 1: work
        self.time_to_next_journey = 0       # 0 - 95

        # Fixed States per Episode
        self.battery_capacity = 0           # 0: 40 kWh, 1: 60 kWh, 2: 80 kWh
        self.energy_per_journey_minute = 0  # 0 - 0.4 kWh per minute
        self.home_charge_power = 0          # 0: 3 kW, 1: 7.4 kW, 2: 11 kW
        self.work_charge_power = 0          # 0: 7.4 kW, 1: 11 kW, 2: 22 kW
        self.out_start_timestep = 0         # 0 - 95
        self.out_duration = 0               # 0 - 95
        self.return_start_timestep = 0      # 0 - 95
        self.return_duration = 0            # 0 - 95

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

        # Measures
        self.accumulated_profit = 0.0

        # --- DEFINE SPACES ---

        # 1. Observation Space

        self.observation_space = spaces.Dict({
            'timestep': spaces.Discrete(96),                                                # 15-minute intervals in a day
            'soc': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),             # State of Charge
            'soc_target': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),      # Target State of Charge
            'energy_price': spaces.Box(low=0.0, high=0.47, shape=(1,), dtype=np.float32),   # Energy price in pounds
            'battery_capacity': spaces.Discrete(3),                                         # Battery capacity in kWh (40 / 60 / 80 kWh)
            'time_to_next_journey': spaces.Discrete(96),                                    # Time to next journey start in timesteps
            'home_charge_power': spaces.Discrete(3),                                        # Home charger power in kW (3 / 7.4 / 11 kW)
            'work_charge_power': spaces.Discrete(3),                                        # Work charger power in kW (7.4 / 11 / 22 kW)
            'location': spaces.Discrete(2)                                                  # home, work
        })

        # 2. Action Space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)  # Continuous action: -1 (discharge) to 1 (charge)

    def __get_obs(self):
        """ 
        Get observations as a dictionary 
        """

        return {
            'timestep': self.timestep,
            'soc': np.array([self.soc], dtype=np.float32),
            'soc_target': np.array([self.soc_target], dtype=np.float32),
            'energy_price': np.array([self.energy_price_profile[self.timestep]], dtype=np.float32),
            'battery_capacity': self.battery_capacity,
            'time_to_next_journey': self.time_to_next_journey,
            'home_charge_power': self.home_charge_power,
            'work_charge_power': self.work_charge_power,
            'location': self.location
        }

    def reset(self, seed=None, options=None):
        """
        Start a new episode

        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional options for environment reset.

        Returns:
            observation (dict): Initial observation of the environment.
            info (dict): Additional information.

        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # --- INITIALISE STATES ---

        # FIXED STATES
        self.battery_capacity = np.random.choice([0, 1, 2])                             # 0: 40 kWh, 1: 60 kWh, 2: 80 kWh
        self.energy_per_journey_minute = np.random.gamma(9.2595, 0.017366)              # Gamma ~ (9.2595, 0.017366)
        self.energy_per_journey_minute = np.clip(self.energy_per_journey_minute, 0, 1.0)

        self.home_charge_power = np.random.choice([0, 1, 2])                            # 0: 3 kW, 1: 7.4 kW, 2: 11 kW
        self.work_charge_power = np.random.choice([0, 1, 2])                            # 0: 7.4 kW, 1: 11 kW, 2: 22 kW
        self.out_start_timestep = math.floor(np.random.gamma(11.4837, 2.61428))         # Gamma ~ (11.4837, 2.61428)
        self.out_start_timestep = np.clip(self.out_start_timestep, 0, 95)

        self.out_duration = math.floor(np.random.gamma(5.9455, 0.46844))                # Gamma ~ (5.9455, 0.46844)
        self.out_duration = np.clip(self.out_duration, 0, 95 - self.out_start_timestep)

        self.return_start_timestep = math.floor(np.random.gamma(24.6641, 1.8946) + 20)  # Gamma ~ (24.6641, 1.8946)
        self.return_start_timestep = np.clip(self.return_start_timestep, self.out_start_timestep + self.out_duration, 95)

        self.return_duration = math.floor(np.random.gamma(5.9455, 0.46844))             # Gamma ~ (5.9455, 0.46844)
        self.return_duration = np.clip(self.return_duration, 0, 95 - self.return_start_timestep)

        # VARIABLE STATES
        self.timestep = 0
        self.day_stage = 0                                                              # Start before work

        self.soc = max(0.0, np.random.normal(loc = 0.4944, scale = 0.1481))             # N ~ (0.4944, 0.1481)
        self.soc = np.clip(self.soc, 0.0, 1.0)

        self.location = 0                                                               # Start at home
        self.soc_target = (self.energy_per_journey_minute * self.out_duration * 1.5) / (self.battery_capacity_map[self.battery_capacity])  # Target SoC to cover outgoing journey
        self.energy_price = self.energy_price_profile[self.timestep]
        self.time_to_next_journey = self.out_start_timestep

        self.accumulated_profit = 0.0

        observation = self.__get_obs()
        info = {}

        return observation, info
    
    def step(self, action):
        """
        Take an action in the environment

        Args:
            action (float): Continuous action between -1 (discharge) and 1 (charge)

        Returns:
            observation (dict): Next observation of the environment.
            reward (float): Reward obtained from the action.
            done (bool): Whether the episode has ended.
            info (dict): Additional information.

        """
        action = np.clip(action, -1.0, 1.0)[0]     

        terminated = False   

        truncated = False

        # --- DUMMY REWARD VARIABLES ---
        deadline_time = False
        missed_target = False
        towed = False
        successful_journey = False

        # --- CALCULATE CHARGING/DISCHARGING ---

        battery_cap = self.battery_capacity_map[self.battery_capacity]
        if self.location == 0:  # At home
            max_charge_power = self.home_charge_power_map[self.home_charge_power]
        else:                   # At work
            max_charge_power = self.work_charge_power_map[self.work_charge_power]

        charge_amount = action * battery_cap

        timesteps_needed = math.ceil(abs(charge_amount) / (max_charge_power * 0.25))

        # --- UPDATE TIMESTEP AND SOC ---

        # No charging/discharging
        if action == 0.0:
            self.timestep += 1

        else:
            # Checks depend on day stage
            deadline_timestep = None
            match self.day_stage:
                case 0:  # Before work
                    deadline_timestep = self.out_start_timestep
                case 1:  # At work
                    deadline_timestep = self.return_start_timestep
                case 2:  # After work
                    deadline_timestep = 95
            
            # If charging time needed exceeds deadline, charge as much as possible
            if self.timestep + timesteps_needed > deadline_timestep:
                available_timesteps = deadline_timestep - self.timestep
                actual_charge_amount = np.sign(charge_amount) * min(abs(charge_amount), max_charge_power * 0.25 * available_timesteps)
                self.soc += actual_charge_amount / battery_cap
                self.timestep = deadline_timestep
            else:
                self.soc += charge_amount / battery_cap
                self.timestep += timesteps_needed

            # Ensure SoC is within bounds
            self.soc = np.clip(self.soc, 0.0, 1.0)
        
        # --- UPDATE LOCATION AND DAY STAGE ---
        if self.day_stage == 0 and self.timestep >= self.out_start_timestep:

            # --- DUMMY REWARD FLAGS ---
            deadline_time = True
            if self.soc < self.soc_target:
                missed_target = True
            else:
                successful_journey = True
            if self.soc < (self.energy_per_journey_minute * self.out_duration) / battery_cap:
                towed = True

            self.location = 1  # Move to work
            self.day_stage = 1
            self.timestep = self.out_start_timestep + self.out_duration
            self.time_to_next_journey = self.return_start_timestep - self.timestep
            self.soc_target = (self.return_duration * self.energy_per_journey_minute * 1.5) / battery_cap

            # Deduct energy for out journey
            energy_used = (self.energy_per_journey_minute * self.out_duration) / battery_cap
            self.soc = max(0.0, self.soc - energy_used)

        elif self.day_stage == 1 and self.timestep >= self.return_start_timestep:

            # --- DUMMY REWARD FLAGS ---
            deadline_time = True
            if self.soc < self.soc_target:
                missed_target = True
            else:
                successful_journey = True
            if self.soc < (self.energy_per_journey_minute * self.return_duration) / battery_cap:
                towed = True

            self.location = 0  # Move to home
            self.day_stage = 2
            self.timestep = self.return_start_timestep + self.return_duration
            self.time_to_next_journey = 96 - self.timestep
            self.soc_target = 0.8 # Target SoC after return journey

            # Deduct energy for return journey
            energy_used = (self.energy_per_journey_minute * self.return_duration) / battery_cap
            self.soc = max(0.0, self.soc - energy_used)

        elif self.day_stage == 2 and self.timestep >= 95:
            terminated = True

        else:
            self.time_to_next_journey = max(0, self.time_to_next_journey - (timesteps_needed if action != 0.0 else 1))

        # --- CALCULATE PROFITS ---
        energy_price = self.energy_price_profile[min(self.timestep, 95)]
        self.accumulated_profit += -action * battery_cap * energy_price

        # --- TODO: REWARD ---
        reward = 0.0

        if deadline_time:
            if missed_target:
                reward -= 50.0
            if towed:
                reward -= 100.0
            if successful_journey:
                reward += 20.0
        else:
            reward += -action * battery_cap * energy_price * 0.1  # Small reward for profit
        
        # --- GET NEXT OBSERVATION ---
        observation = self.__get_obs()

        info = {
            'accumulated_profit': self.accumulated_profit
        }

        if not self.observation_space.contains(observation):
            print("INVALID OBSERVATION!")
            print("obs:", observation)
            print("obs type:", type(observation))
            print("obs dtype:", getattr(observation, "dtype", None))
            print("obs shape:", getattr(observation, "shape", None))
            print("space:", self.observation_space)
            raise ValueError("Observation not in observation_space")

        return observation, reward, terminated, truncated, info       