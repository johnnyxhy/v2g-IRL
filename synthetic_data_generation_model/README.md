**EV Charging - Synthetic Data Generation Model**


**Key features:**

- One day of data for each of 500 individuals, showing driving, parking, charging and discharging activities across 96 15-minute timesteps. Driving patterns are included exogenously, taken from UK National Travel Survey 2023 NTS metadata.

- All the individuals are commuters, each travelling from home to work and back during the day. The journey timings, distances and average driving speeds for these commutes vary and come from the NTS data.
  
- The NTS data was filtered to only include weekday journeys of 10 miles or more and to include 17-20, 21-29, 30-39, 40-49, 50-59 and 60+ age categories, for both males and females, retaining the original % distribution across these segments.
  
- Three EV battery capacities of 40, 60 and 80 kWh are assigned at random to each individual. These are broadly in keeping with battery sizes of EVs in the market.
  
- Three charge rates are included for home (3, 7.4 and 11 kW) and work (7.4, 11 and 22 kW) chargers, which are also assigned at random to each EV user. These are based on speeds of chargers currently available in the market.
  
- Each EV has an initial energy level assigned at random, based on a distribution with a mean charge level of 50% and standard deviation of 15%. Energy is added to or depleted during charging and discharging, dependent on charge rates. It also depletes at 0.33 kWh/mile during driving.

- A maximum charge level of 80% of battery capacity is assumed and factored in as the limit in any charge decisions. A minimum level of 20% of battery capacity is also assumed and taken to be the lower limit when discharging. Such upper and lower limits are often used by EV users and factored into EV charging settings.
  
- Upcoming journey energy requirements are calculated based on the journey distances assigned from the NTS data. Target energy is determined by adding minimum charge level to upcoming trip energy requirement * 1.5, assuming that the commuters want to build in some range buffer.
  
- Energy price is the same across charge points and set with an average level of £0.27 per kWh [44]. The price then varies based on a sine wave with amplitude of £0.20. This variability is intended to drive differences in charging behaviours.
  
- Agents make charge, discharge or 'do nothing' decisions in any time steps where they are not driving or charging. Agents calculate the utility of each of these actions via a utility function and then select actions from a probability distribution, using a multinomial logit (MNL) approach.
  
- This model includes a discrete action space, meaning that utilities are calculated for charging and discharging actions that charge or discharge 10%, 20%, 30%, 40% or 50% of battery capacity. 

**Utility functions:**

- Charging utility = (Beta_0_charge + (Beta_1_charge * Utility_Energy_Level) + (Beta_2_charge * Utility_Price)

- Discharging utility = (Beta_0_discharge + (Beta_1_discharge * Utility_Energy_Level) + (Beta_2_discharge * - Utility_Price)

- None (no action) = beta0_none 

Where:

**Utility_Energy_Level =**

* For each charge or discharge action, if there is a shortfall in expected energy % post charge action vs the target energy level %, utility is calculated as e^(-15 * the energy gap).

* If there is a surplus in expected energy % post charge action vs the target energy level %, utility is calculated as e^(-8 * the energy gap).

- Energy gap = (energy post charge - required energy) / battery capacity<br>
- Energy post charge = current state of charge + charge amount<br>
- Required energy = minimum charge level + trip energy requirement * buffer<br>
- Minimum charge level = 20% of battery capacity in all cases. Buffer = 1.5 in all cases<br>

- The above means non-linearity in how shortfalls and surpluses are perceived. People are risk averse around the shortfalls, with a steeper drop off in utility for shortfalls below the target level vs a gentler drop off in utility for surpluses above the target level.

**Utility_Price =**

- The prices have been scaled linearly from a utility of 1 for the lowest price (0.07) to -1 for the highest price (0.47). The discharge price utility has a negative sign, meaning it is the opposite of charge utility. E.g. When charge utility = 1, discharge utility = -1.

- Currently, beta_0_charge, beta_0_discharge and beta0_none are all set to zero. If non-zero Beta_0 values are added in, these are 'alternative specific constant' (ASC) values, reflecting any inherent preferences for options over a reference option, before accounting attribute utilities.

**Betas:**
  
- The other betas are currently set as follows, reflecting varying attitudes to energy level and price across the EV user segments:

segment_parameters = {
    "Female 17-20": {"beta1_charge": 0.85, "beta2_charge": 0.3, "beta1_discharge": 0.85, "beta2_discharge": 0.3},
    "Male 17-20": {"beta1_charge": 0.9, "beta2_charge": 0.3, "beta1_discharge": 0.9, "beta2_discharge": 0.3},
    "Female 21-29": {"beta1_charge": 0.85, "beta2_charge": 0.3, "beta1_discharge": 0.85, "beta2_discharge": 0.3},
    "Male 21-29": {"beta1_charge": 0.9, "beta2_charge": 0.3, "beta1_discharge": 0.9, "beta2_discharge": 0.3},
    "Female 30-39": {"beta1_charge": 0.75, "beta2_charge": 0.15, "beta1_discharge": 0.75, "beta2_discharge": 0.15},
    "Male 30-39": {"beta1_charge": 0.8, "beta2_charge": 0.15, "beta1_discharge": 0.8, "beta2_discharge": 0.15},
    "Female 40-49": {"beta1_charge": 0.75, "beta2_charge": 0.15, "beta1_discharge": 0.75, "beta2_discharge": 0.15},
    "Male 40-49": {"beta1_charge": 0.8, "beta2_charge": 0.15, "beta1_discharge": 0.8, "beta2_discharge": 0.15},
    "Female 50-59": {"beta1_charge": 0.65, "beta2_charge": 0.3, "beta1_discharge": 0.65, "beta2_discharge": 0.3},
    "Male 50-59": {"beta1_charge": 0.7, "beta2_charge": 0.3, "beta1_discharge": 0.7, "beta2_discharge": 0.3},
    "Female 60+": {"beta1_charge": 0.65, "beta2_charge": 0.3, "beta1_discharge": 0.65, "beta2_discharge": 0.3},
    "Male 60+": {"beta1_charge": 0.7, "beta2_charge": 0.3, "beta1_discharge": 0.7, "beta2_discharge": 0.3}
}

- There are rules built in, meaning that charge and discharge actions are unavailable for selection if they would result taking state of charge below the min or max levels (set at 20% and 80%).

**Probabilistic action selection:**

- Feasible actions are assigned probabilities via the following standard multinomial logit (MNL) formula (assuming. As an example, the probability of selecting action 1 (A1) from 4 possible charging actions (A1, A2, A3 and A4), can be calculated as follows using the MNL method:

- Probability of selection A1 = e^A1_Utility / (e^A1_Utility + e^A2_Utility + e^A3_Utility + e^A4_Utility).

- Probabilities can be calculated in the same way for each possible action. An action is then selected randomly from across the probability distribution.
