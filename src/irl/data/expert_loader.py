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

    # --- 3. Convert Trip times to timestep ---
    df['out_start_timestep'] = (df['Trip_Start_Time_Out_Mins'] / 15).astype(int)
    df['out_end_timestep'] = (df['Trip_End_Time_Out_Mins'] / 15).astype(int)
    df['return_start_timestep'] = (df['Trip_Start_Time_Return_Mins'] / 15).astype(int)
    df['return_end_timestep'] = (df['Trip_End_Time_Return_Mins'] / 15).astype(int)

    # --- 4. Convert Energy to SoC ---
    df['SoC'] = df['Battery_Energy_Level_kWh'] / df['Battery_Capacity_kWh']
    df['SoC_target'] = np.where(df['Target_Energy_kWh'].isna(), 0.0, df['Target_Energy_kWh'] / df['Battery_Capacity_kWh'])

    # --- 5. Convert Charge/Discharge amount to SoC ---
    df['Charge_Amount_SoC'] = df['Total_Charge_kWh'] / df['Battery_Capacity_kWh']
    df['Discharge_Amount_SoC'] = df['Total_Discharge_kWh'] / df['Battery_Capacity_kWh']
    df['Charge_Action'] = np.select([df['Action'] == 'charge', df['Action'] == 'discharge'], 
                                    [df['Charge_Amount_SoC'], -df['Discharge_Amount_SoC']], 
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

        # --- EXTRACT OBSERVATIONS ---

        # ORDER: [timestep, soc, soc_target, energy_price, soc_initial, battery_capacity, out_start, out_end, return_start, return_end, home_charge_power, work_charge_power, location]

        # --- Extract continuous values ---
        continuous_obs = episode_data[[
            'Timestep',
            'SoC',
            'SoC_target',
            'Energy_Price_Pounds',
            'Initial Energy Percent',
            'Battery_Capacity_kWh',
            'out_start_timestep',
            'out_end_timestep',
            'return_start_timestep',
            'return_end_timestep',
            'Home_Charger_kW',
            'Work_Charger_kW'
        ]].values.astype(np.float64)    

        # --- Extract location as one-hot encoding ---
        loc_indices = episode_data['loc_int'].values
        loc_onehot = np.eye(5)[loc_indices].astype(np.float64)

        # --- Combine continuous and one-hot observations ---
        observations = np.hstack([continuous_obs, loc_onehot])

        # --- EXTRACT ACTIONS ---
        actions = episode_data['Charge_Action'].values.astype(np.float64)

        episodes.append({
            'episodeID': episode_id,
            'segment': episode_data['Segment'].iloc[0],
            'observations': observations.tolist(),
            'actions': actions.tolist()
        })

    # Check if energy price array in every episode is exactly the same 
    first_episode_prices = episodes[0]['observations']
    for ep in episodes[1:]:
        if not np.array_equal(np.array(first_episode_prices)[:,3], np.array(ep['observations'])[:,3]):
            print("Warning: Energy price profiles differ between episodes.")
            break
    print("All episodes have consistent energy price profiles.")

    # --- SAVE TO JSON ---
    print(f"Extracted {len(episodes)} valid episodes, saving to JSON...")
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(episodes, f, indent=4)
    print("Data loading and preprocessing complete.")

    return episodes


# Example usage:
episodes = load_trajectories("data/EVdataset.csv", output_file="data/processed_trajectories.json")