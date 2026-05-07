import pandas as pd
import numpy as np
import json

def load_trajectories(input_file, output_file=None):
    """
    Load trajectory data from a CSV file and preprocess it into episodes 
    suitable for Deep MaxEnt IRL. Extracts per-action state-action pairs 
    (raw observations + actions) alongside feature expectations for monitoring.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str, optional): Path to save the processed JSON file.
        
    Returns:
        list: A list of processed episodes with state_action_pairs.
    """

    try:
        df = pd.read_csv(input_file)
    except ValueError:
        print("Error loading CSV file.")
        return None

    # --- DATA PREPROCESSING ---

    # 1. Create EpisodeIDs
    df['EpisodeID'] = (df['Timestep'] == 0).cumsum()

    # 2. Map Locations
    loc_map = {'home': 0, 'work': 1, 'driving_out': 2, 'driving_return': 3, 'towed': 4}
    df['loc_int'] = df['Location'].map(loc_map)

    # 3. Find journey timesteps and duration
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

    # 4. Time to next journey
    cond_start_home = df['Timestep'] < df['out_start_timestep']
    cond_work = (df['Timestep'] >= df['out_start_timestep']) & (df['Timestep'] < df['return_start_timestep'])
    cond_end_home = df['Timestep'] >= df['return_start_timestep']
    df['timesteps_to_next_journey'] = np.select(
        [cond_start_home, cond_work, cond_end_home],
        [df['out_start_timestep'] - df['Timestep'], df['return_start_timestep'] - df['Timestep'], 96 - df['Timestep']],
        default=0
    )

    # 5. Convert Energy to SoC
    df['SoC_end'] = df['Battery_Energy_Level_kWh'] / df['Battery_Capacity_kWh']
    df['SoC_target'] = np.where(df['Upcoming_Trip_Energy_kWh'].isna(), 0.0, df['Upcoming_Trip_Energy_kWh'] / df['Battery_Capacity_kWh']) + 0.2
    #df['SoC_target'] = df['Upcoming_Energy_Requirement_%']

    # 6. SoC at start of timestep
    df['SoC'] = df.groupby('EpisodeID')['SoC_end'].shift(1)
    first_timesteps = df['Timestep'] == 0
    df.loc[first_timesteps, 'SoC'] = df.loc[first_timesteps, 'Initial_Energy_kWh'] / df.loc[first_timesteps, 'Battery_Capacity_kWh']
    df['SoC_gap'] = df['SoC'] - df['SoC_target']

    # 7. Charge/discharge amounts as SoC fractions
    df['amount_charged'] = (df['Total_Charge_kWh'] / df['Battery_Capacity_kWh']).fillna(0.0)
    df['amount_discharged'] = (df['Total_Discharge_kWh'] / df['Battery_Capacity_kWh']).fillna(0.0)

    # 8. Map battery/charger to indices
    df['battery_cap_index'] = df['Battery_Capacity_kWh'].map({40: 0, 60: 1, 80: 2})
    df['home_charge_index'] = df['Home_Charger_kW'].map({3: 0, 7.4: 1, 11: 2})
    df['work_charge_index'] = df['Work_Charger_kW'].map({7.4: 0, 11: 1, 22: 2})

    # 9. Target tracking features (for monitoring feature expectations)
    df['battery_needed_target'] = (np.maximum(0.0, df['SoC_target'] - df['SoC_end'])) ** 2
    df['battery_exceeded_target'] = (np.maximum(0.0, df['SoC_end'] - df['SoC_target'])) ** 2

    # 10. Journey failure
    df['journey_failure'] = np.where(
        (df['Location'].shift(1).isin(['driving_out', 'driving_return']) & (df['SoC_end'].shift(1) <= 0.0)),
        1.0, 0.0
    )
    df['journey_failure'] = df['journey_failure'].fillna(0.0)

    # Global price stats
    mean_price = df['Energy_Price_Pounds'].mean()
    max_price = df['Energy_Price_Pounds'].max()

    # --- EXTRACT EPISODES ---
    print(f"Extracting episodes...")
    episodes = []

    for episode_id, episode_data in df.groupby('EpisodeID'):
        if len(episode_data) != 96:
            continue
        episode_data = episode_data.sort_values(by='Timestep').reset_index(drop=True)
        if (episode_data['out_duration'].iloc[0] == 0) or (episode_data['return_duration'].iloc[0] == 0):
            continue

        # Full SoC history (all 96 timesteps)
        soc_history = episode_data['SoC'].tolist()

        # Filter to home/work segments only
        episode_data = episode_data[episode_data['Location'].isin(['home', 'work'])].reset_index(drop=True)

        # Initial values (same as linear loader)
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

        # --- Identify action boundaries ---
        action_starts = (episode_data.index == 0) | (episode_data['Action_Duration_Timesteps'].shift(1) == 0)
        episode_data['action_group'] = action_starts.cumsum()

        # --- Compute feature expectations (for monitoring, same as linear loader) ---
        episode_data['has_transfer'] = (episode_data['Amount_Charged_During_Timestep_kWh'].abs() > 0).astype(float)
        episode_data['price_x_transfer'] = episode_data['Energy_Price_Pounds'] * episode_data['has_transfer']
        grp = episode_data.groupby('action_group')
        action_mean_price = grp['price_x_transfer'].transform('sum') / grp['has_transfer'].transform('sum').replace(0, np.nan)
        action_mean_price = action_mean_price.fillna(mean_price)

        episode_data['charge_cost_penalty'] = 10.0 * episode_data['amount_charged'] ** 2 * (action_mean_price / max_price)
        episode_data['discharge_cost_penalty'] = 10.0 * episode_data['amount_discharged'] ** 2 * ((max_price - action_mean_price) / max_price)

        action_dt = grp['Timestep'].transform('count')
        episode_data['action_dt'] = action_dt  # store per-row delta_t for IRL weighting
        episode_data['battery_needed_target'] = episode_data['battery_needed_target'] * action_dt
        episode_data['battery_exceeded_target'] = episode_data['battery_exceeded_target'] * action_dt

        # Filter to action-start rows only
        action_start_rows = episode_data[action_starts].reset_index(drop=True)

        # --- Feature expectations (summed over actions, for monitoring) ---
        features = action_start_rows[[
            'amount_charged', 'amount_discharged',
            'charge_cost_penalty', 'discharge_cost_penalty',
            'battery_needed_target', 'battery_exceeded_target',
            'journey_failure',
        ]].values.astype(np.float64)
        feature_expectation = np.sum(features, axis=0)

        # --- Extract state-action pairs (raw, unnormalized) ---
        observations = []
        actions = []
        delta_ts = []

        for _, row in action_start_rows.iterrows():
            # Observation: [timestep, soc, soc_gap, energy_price, battery_cap_index, time_to_next_journey, charger_power_kW]
            if row['Location'] == 'home':
                charger_power = row['Home_Charger_kW']
            else:
                charger_power = row['Work_Charger_kW']

            obs = [
                float(row['Timestep']),
                float(row['SoC']),
                float(row['SoC_gap']),
                float(row['Energy_Price_Pounds']),
                float(row['battery_cap_index']),
                float(row['timesteps_to_next_journey']),
                float(charger_power),
            ]
            observations.append(obs)

            # Action: discrete index 0-20 (10 = idle, >10 = charge, <10 = discharge)
            # Stored as normalized float in [-1, 1]: (discrete - 10) / 10
            # This matches how the env normalizes action_for_reward before the reward net.
            ACTION_STEP_SIZE = 0.1
            charge_soc = row['amount_charged']
            discharge_soc = row['amount_discharged']
            if charge_soc > 0:
                discrete_val = round(charge_soc / ACTION_STEP_SIZE, 2) + 10
            elif discharge_soc > 0:
                discrete_val = round(10 - discharge_soc / ACTION_STEP_SIZE, 2)
            else:
                discrete_val = 10.0
            action_val = (discrete_val - 10.0) / 10.0  # normalize to [-1, 1]

            actions.append([action_val])
            delta_ts.append(int(row['action_dt']))

        episodes.append({
            'episodeID': int(episode_id),
            'segment': episode_data['Segment'].iloc[0],
            'initial_values': initial_values,
            'soc_history': soc_history,
            'feature_expectation': feature_expectation.tolist(),
            'state_action_pairs': {
                'observations': observations,
                'actions': actions,
                'delta_ts': delta_ts,
            },
        })

    print(f"Extracted {len(episodes)} valid episodes.")

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(episodes, f, indent=4)
        print(f"Saved to {output_file}")

    return episodes


if __name__ == "__main__":
    episodes = load_trajectories(
        "data/EVDataset_discrete_pricediff.csv",
        output_file="data/processed_trajectories_deep_discrete_gap_pricediff.json"
    )
