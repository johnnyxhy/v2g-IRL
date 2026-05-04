import pandas as pd
import numpy as np
import json

def load_trajectories(input_file, output_file=None):
    """
    Load trajectory data from a CSV file and preprocess it into episodes suitable for IRL. Saves as JSON if output_file is provided.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str, optional): Path to save the processed JSON file. If None, does not save.
        
    Returns:
        list: A list of processed episodes.
    """

    try:
        df = pd.read_csv(input_file)
    except ValueError:
        print("Error loading CSV file.")
        return None

    # --- DATA PREPROCESSING ---

    # --- 1. Create EpisodeIDs since each individual has multiple trajectories ---
    df['EpisodeID'] = (df['Timestep'] == 0).cumsum()

    # --- 2. Map Locations to discrete integer values ---
    loc_map = {
        'home': 0,
        'work': 1,
        'driving_out': 2,
        'driving_return': 3,
        'towed': 4
    }
    df['loc_int'] = df['Location'].map(loc_map)

    # --- 3. Find journey timesteps and duration

    df['out_start_timestep'] = df.groupby('EpisodeID')['Timestep'].transform(
        lambda x: x[df.loc[x.index, 'Location'] == 'driving_out'].min() if any(df.loc[x.index, 'Location'] == 'driving_out') else 96
    )
    df['return_start_timestep'] = df.groupby('EpisodeID')['Timestep'].transform(
        lambda x: x[df.loc[x.index, 'Location'] == 'driving_return'].min() if any(df.loc[x.index, 'Location'] == 'driving_return') else 96
    )
    df['out_duration'] = df.groupby('EpisodeID')['Timestep'].transform(
        lambda x: x[df.loc[x.index, 'Location'] == 'driving_out'].count() if any(df.loc[x.index, 'Location'] == 'driving_out') else 0
    )
    df['return_duration'] = df.groupby('EpisodeID')['Timestep'].transform(
        lambda x: x[df.loc[x.index, 'Location'] == 'driving_return'].count() if any(df.loc[x.index, 'Location'] == 'driving_return') else 0
    )

    # --- 4. Find timestep to next journey --- 
    # 1. Define the conditions for the three phases of the day
    cond_start_home = df['Timestep'] < df['out_start_timestep']
    cond_work = (df['Timestep'] >= df['out_start_timestep']) & (df['Timestep'] < df['return_start_timestep'])
    cond_end_home = df['Timestep'] >= df['return_start_timestep']

    # 2. Define the calculations for each phase
    calc_start_home = df['out_start_timestep'] - df['Timestep']
    calc_work = df['return_start_timestep'] - df['Timestep']
    calc_end_home = 96 - df['Timestep']

    # 3. Apply the logic using np.select
    df['timesteps_to_next_journey'] = np.select(
        [cond_start_home, cond_work, cond_end_home],  # The conditions
        [calc_start_home, calc_work, calc_end_home],  # The matching values
        default=0
    )
    # --- 5. Convert Energy to SoC ---
    df['SoC_end'] = df['Battery_Energy_Level_kWh'] / df['Battery_Capacity_kWh'] # This value is SOC at end of timestep
    df['SoC_target'] = np.where(df['Upcoming_Trip_Energy_kWh'].isna(), 0.0, df['Upcoming_Trip_Energy_kWh'] / df['Battery_Capacity_kWh'])

    # --- 6. Find SoC at start of timestep ---
    df['SoC'] = df.groupby('EpisodeID')['SoC_end'].shift(1)
    # first timestep SoC is initial SoC
    first_timesteps = df['Timestep'] == 0
    df.loc[first_timesteps, 'SoC'] = df.loc[first_timesteps, 'Initial_Energy_kWh'] / df.loc[first_timesteps, 'Battery_Capacity_kWh']

    # --- 7. Convert Total Charge to Percentage of Battery Capacity ---
    df['amount_charged'] = (df['Total_Charge_kWh'] / df['Battery_Capacity_kWh']).fillna(0.0) 
    df['amount_discharged'] = (df['Total_Discharge_kWh'] / df['Battery_Capacity_kWh']).fillna(0.0)
    
    # --- 8. Map Battery Capacity and Charging Power to index --- 
    battery_cap_map = {40: 0, 60: 1, 80: 2}
    df['battery_cap_index'] = df['Battery_Capacity_kWh'].map(battery_cap_map)
    home_charge_map = {3: 0, 7.4: 1, 11: 2}
    df['home_charge_index'] = df['Home_Charger_kW'].map(home_charge_map)
    work_charge_map = {7.4: 0, 11: 1, 22: 2}
    df['work_charge_index'] = df['Work_Charger_kW'].map(work_charge_map)
    
    # --- 10. Battery needed and exceeded target ---
    df['battery_needed_target'] = (np.maximum(0.0, df['SoC_target'] - df['SoC_end'])) ** 2 * (df['Action_Duration_Timesteps']+1)
    df['battery_exceeded_target'] = (np.maximum(0.0, df['SoC_end'] - df['SoC_target'])) ** 2 * (df['Action_Duration_Timesteps']+1)

    # --- 12. Journey failure ---
    # Decided if in row above, location is driving_out or driving_return and SoC_end == 0
    df['journey_failure'] = np.where(
        (
            df['Location'].shift(1).isin(['driving_out', 'driving_return']) &
            (df['SoC_end'].shift(1) <= 0.0)
        ),
        1.0,
        0.0
    )
    df['journey_failure'] = df['journey_failure'].fillna(0.0)

    # --- 13. Charge Action Taken --
    df['charge_action_taken'] = np.where(
        (df['amount_charged'] > 0) | (df['amount_discharged'] > 0),
        1.0/96.0,  # Scale by max possible amount of charge/discharge in one timestep to keep in range [0,1]
        0
    )

    # --- EXTRACT EPISODES ---
    grouped = df.groupby('EpisodeID')
    print(f"Extracting {len(grouped)} episodes...")
    episodes = []
    for episode_id, episode_data in df.groupby('EpisodeID'):

        # --- ENSURE EPISODE LENGTH ---
        if len(episode_data) != 96:
            continue

        # --- ENSURE SORTED BY TIMESTEP ---
        episode_data = episode_data.sort_values(by='Timestep').reset_index(drop=True)

        # --- IGNORE EPISODES WITH 0 OUT OR RETURN DURATIONS ---
        if (episode_data['out_duration'].iloc[0] == 0) or (episode_data['return_duration'].iloc[0] == 0):
            continue

        # --- EXTRACT FULL SOC HISTORY ---
        soc_history = episode_data['SoC'].tolist()

        # --- ONLY KEEP HOME AND WORK SEGMENTS ---
        episode_data = episode_data[episode_data['Location'].isin(['home', 'work'])].reset_index(drop=True)

        # --- RECOMPUTE battery_needed/exceeded_target USING END-OF-ACTION SoC ---
        # Currently SoC_end on each row is per-timestep. We want the SoC at the end
        # of the entire action (last timestep), to match how the env evaluates features.
        action_starts = (episode_data.index == 0) | (episode_data['Action_Duration_Timesteps'].shift(1) == 0)
        action_group = action_starts.cumsum()
        soc_end_of_action = episode_data.groupby(action_group)['SoC_end'].transform('last')
        action_duration = episode_data.groupby(action_group)['SoC_end'].transform('size')

        episode_data['battery_needed_target'] = (np.maximum(0.0, episode_data['SoC_target'] - soc_end_of_action)) ** 2 * action_duration
        episode_data['battery_exceeded_target'] = (np.maximum(0.0, soc_end_of_action - episode_data['SoC_target'])) ** 2 * action_duration

        # --- EXTRACT INITIAL VALUES ---
        initial_values = {
            'soc': episode_data['SoC'].iloc[0].item(),
            'battery_capacity': episode_data['battery_cap_index'].iloc[0].item(),
            'journey_distance': episode_data['Upcoming_Trip_Distance_Miles'].iloc[0].item(),
            'out_journey_speed': episode_data['Average_Speed_Out_mph'].iloc[0].item(),
            'return_journey_speed': episode_data['Average_Speed_Return_mph'].iloc[0].item(),
            'out_start_timestep': episode_data['out_start_timestep'].iloc[0].item(),
            'return_start_timestep': episode_data['return_start_timestep'].iloc[0].item(),
            'home_charge_power': episode_data['home_charge_index'].iloc[0].item(),
            'work_charge_power': episode_data['work_charge_index'].iloc[0].item(),
        }

        # --- Extract only for first entry and when Action_Duration_Timesteps is zero for the row above --- 
        episode_data = episode_data[(episode_data.index == 0) | (episode_data['Action_Duration_Timesteps'].shift(1) == 0)].reset_index(drop=True)

        # --- EXTRACT FEATURES ---
        # [amount_charged, amount_discharged, battery_needed_target, battery_exceeded_target, soc_outside_range, journey_failure]

        features = episode_data[[
            'amount_charged',
            'amount_discharged',
            'battery_needed_target',
            'battery_exceeded_target',
            'journey_failure',
        ]].values.astype(np.float64)

        # Calculate feature expectations 
        feature_expectation = np.mean(features, axis=0)

        # --- EXTRACT OBSERVATIONS ---

        # ORDER: [timestep, soc, soc_target, energy_price, battery_capacity, out_start, out_end, return_start, return_end, home_charge_power, work_charge_power, location]

        # # --- Extract continuous values ---
        # continuous_obs = episode_data[[
        #     'Timestep',
        #     'SoC',
        #     'SoC_target',
        #     'Energy_Price_Pounds',
        #     'battery_cap_index',
        #     'timesteps_to_next_journey',
        #     'home_charge_index',
        #     'work_charge_index'
        # ]].values.astype(np.float64)    

        # # --- Extract location as one-hot encoding ---
        # loc_indices = episode_data['loc_int'].values
        # loc_onehot = np.eye(2)[loc_indices].astype(np.float64)

        # # --- Combine continuous and one-hot observations ---
        # observations = np.hstack([continuous_obs, loc_onehot])

        # # --- EXTRACT ACTIONS ---
        # actions = episode_data['charge_action_index'].values.astype(np.int64)

        episodes.append({
            'episodeID': episode_id,
            'segment': episode_data['Segment'].iloc[0],
            'feature_expectation': feature_expectation.tolist(),
            'features': features.tolist(),
            'soc_history': soc_history,
            'initial_values': initial_values,
            #'observations': observations.tolist(),
            #'actions': actions.tolist()
        })

    # --- SAVE TO JSON ---
    print(f"Extracted {len(episodes)} valid episodes, saving to JSON...")
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(episodes, f, indent=4)
    print("Data loading and preprocessing complete.")

    return episodes


# Example usage:
episodes = load_trajectories("data/EVdataset_continuous.csv", output_file="data/processed_trajectories_continuous.json")