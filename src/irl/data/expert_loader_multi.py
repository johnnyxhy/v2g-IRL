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
    df['SoC_target'] = np.where(df['Target_Energy_kWh'].isna(), 0.0, df['Target_Energy_kWh'] / df['Battery_Capacity_kWh'])

    # --- 6. Find SoC at start of timestep ---
    df['SoC'] = df.groupby('EpisodeID')['SoC_end'].shift(1)
    # first timestep SoC is initial SoC
    first_timesteps = df['Timestep'] == 0
    df.loc[first_timesteps, 'SoC'] = df.loc[first_timesteps, 'Initial_Energy_kWh'] / df.loc[first_timesteps, 'Battery_Capacity_kWh']

    # --- 7. Convert Total Charge to Percentage of Battery Capacity ---
    df['Charge_Action'] = np.select([df['Action'] == 'charge', df['Action'] == 'discharge', df['Action'] == 'none'],
                                    [df['Total_Charge_kWh'] / df['Battery_Capacity_kWh'],
                                    -df['Total_Discharge_kWh'] / df['Battery_Capacity_kWh'],
                                    0.0],
                                    default=0.0)
    
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

        # --- IGNORE EPISODES WITH 0 OUT OR RETURN DURATIONS -- 
        if (episode_data['out_duration'].iloc[0] == 0) or (episode_data['return_duration'].iloc[0] == 0):
            continue

        # --- ONLY KEEP HOME AND WORK SEGMENTS ---
        episode_data = episode_data[episode_data['Location'].isin(['home', 'work'])].reset_index(drop=True)

        # --- EXTRACT INITIAL VALUES ---
        initial_values = {
            'Initial_SoC': episode_data['SoC'].iloc[0].item(),
            'Battery_Capacity_kWh': episode_data['Battery_Capacity_kWh'].iloc[0].item(),
            'Energy_per_Journey_Minute': episode_data['Energy per journey minute'].iloc[0].item(),
            'Out_Start_Timestep': episode_data['out_start_timestep'].iloc[0].item(),
            'Return_Start_Timestep': episode_data['return_start_timestep'].iloc[0].item(),
            'Out_Duration': episode_data['out_duration'].iloc[0].item(),
            'Return_Duration': episode_data['return_duration'].iloc[0].item(),
            'Home_Charger_kW': episode_data['Home_Charger_kW'].iloc[0].item(),
            'Work_Charger_kW': episode_data['Work_Charger_kW'].iloc[0].item(),
        }

        # --- EXTRACT OBSERVATIONS ---

        # ORDER: [timestep, soc, soc_target, energy_price, battery_capacity, out_start, out_end, return_start, return_end, home_charge_power, work_charge_power, location]

        # --- Extract only when charge action changes ---
        episode_data = episode_data[episode_data['Charge_Action'].diff().fillna(1) != 0].reset_index(drop=True)
        
        # --- Extract continuous values ---
        continuous_obs = episode_data[[
            'Timestep',
            'SoC',
            'SoC_target',
            'Energy_Price_Pounds',
            'Battery_Capacity_kWh',
            'timesteps_to_next_journey',
            'Home_Charger_kW',
            'Work_Charger_kW'
        ]].values.astype(np.float64)    

        # --- Extract location as one-hot encoding ---
        loc_indices = episode_data['loc_int'].values
        loc_onehot = np.eye(2)[loc_indices].astype(np.float64)

        # --- Combine continuous and one-hot observations ---
        observations = np.hstack([continuous_obs, loc_onehot])

        # --- EXTRACT ACTIONS ---
        actions = episode_data['Charge_Action'].values.astype(np.float64)

        episodes.append({
            'episodeID': episode_id,
            'segment': episode_data['Segment'].iloc[0],
            'initial_values': initial_values,
            'observations': observations.tolist(),
            'actions': actions.tolist()
        })

    # --- SAVE TO JSON ---
    print(f"Extracted {len(episodes)} valid episodes, saving to JSON...")
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(episodes, f, indent=4)
    print("Data loading and preprocessing complete.")

    return episodes


# Example usage:
episodes = load_trajectories("data/EVdataset.csv", output_file="data/processed_trajectories_multi.json")