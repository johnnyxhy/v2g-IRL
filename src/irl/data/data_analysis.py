import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

CSV_FILE = 'data/EVDataset.csv'

# --- 1. LOAD DATA ---

df = pd.read_csv(CSV_FILE)

# --- 2. GROUP BY EPISODE ID ---

df['EpisodeID'] = (df['Timestep'] == 0).cumsum()

# --- 3. TRANSFORM DATA ---

# 1. Convert initial energy to SoC
df['SoC_Initial'] = df['Initial_Energy_kWh'] / df['Battery_Capacity_kWh']

# 2. Convert Start and End times to timesteps
df['Out_Start_Timestep'] = (df['Trip_Start_Time_Out_Mins'] / 15)
df['Return_Start_Timestep'] = (df['Trip_Start_Time_Return_Mins'] / 15)

# 3. Convert Journey Times to timestep
df['Trip_Journey_Time_Timesteps'] = (df['Trip Journey Time'] / 15)
df['Return_Journey_Time_Timesteps'] = (df['Return Journey Time'] / 15)

# --- 4. EXTRACT VALUES ---

def extract_values(df, col_name, timestep):
    sub = df.loc[df['Timestep'] == timestep, ['EpisodeID', col_name]].dropna()
    return sub[col_name].values

soc_initial = extract_values(df, 'SoC_Initial', 0)
energy_per_journey_minute = extract_values(df, 'Energy per journey minute', 0)
out_start_timestep = extract_values(df, 'Out_Start_Timestep', 0)
return_start_timestep = extract_values(df, 'Return_Start_Timestep', 0)
trip_journey_time = extract_values(df, 'Trip_Journey_Time_Timesteps', 0)
return_journey_time = extract_values(df, 'Return_Journey_Time_Timesteps', 0)
journey_time = np.concatenate([trip_journey_time, return_journey_time])

# --- 5. PERFORM FITTING ---
def fit_distribution(data, distribution_name, floc = 0):
    distribution = getattr(stats, distribution_name)

    data = data[np.isfinite(data)]
    data = data[~np.isnan(data)]

    if distribution_name == 'norm':
        mu, std = distribution.fit(data)
        params = (mu, std)

    elif distribution_name == 'gamma':
        data_pos = data[data > 0]
        a, loc, scale = distribution.fit(data_pos, floc=floc)
        params = (a, loc, scale)

    return params

# --- 6. VISUALISATION ---
def plot_distribution_fit(data, distribution_name, bins=30, name='Data', floc=0):

    params = fit_distribution(data, distribution_name, floc=floc)

    plt.figure()

    plt.hist(data, bins=bins, density=True, alpha=0.5, color='g', label='Data Histogram')

    data_min, data_max = np.min(data), np.max(data)
    x = np.linspace(data_min, data_max, 500)

    if distribution_name == 'norm':
        mu, sigma = params
        plt.plot(x, stats.norm.pdf(x, mu, sigma), linewidth=2, label=f"Normal (mu={mu:.4f}, sigma={sigma:.4f})")
    elif distribution_name == 'gamma':
        a, loc, scale = params
        plt.plot(x, stats.gamma.pdf(x, a, loc=loc, scale=scale), linewidth=2, label=f"Gamma (a={a:.4f}, loc={loc:.4f}, scale={scale:.4f})")

    plt.title(f'{distribution_name} Distribution Fit for {name}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()

    print(f"Fitted parameters for {name} ({distribution_name}): {params}")

# Manual Option
def plot_distribution_fit_manual(data, distribution_name, bins=30, name='Data', params=None, floc=0):

    plt.figure()

    plt.hist(data, bins=bins, density=True, alpha=0.5, color='g', label='Data Histogram')

    data_min, data_max = np.min(data), np.max(data)
    x = np.linspace(data_min, data_max, 500)

    if distribution_name == 'norm':
        mu, sigma = params
        plt.plot(x, stats.norm.pdf(x, mu, sigma), linewidth=2, label=f"Normal (mu={mu:.4f}, sigma={sigma:.4f})")
    elif distribution_name == 'gamma':
        a, loc, scale = params
        plt.plot(x, stats.gamma.pdf(x, a, loc=loc, scale=scale), linewidth=2, label=f"Gamma (a={a:.4f}, loc={loc:.4f}, scale={scale:.4f})")

    plt.title(f'{distribution_name} Distribution Fit for {name}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()

# --- EXAMPLE USAGE ---
plot_distribution_fit(soc_initial, 'norm', bins=50, name='Initial SoC')
plot_distribution_fit(energy_per_journey_minute, 'gamma', bins=50, name='Energy per Journey Minute')
plot_distribution_fit(out_start_timestep, 'gamma', bins=30, name='Out Start Timestep')
plot_distribution_fit(return_start_timestep, 'gamma', bins=30, name='Return Start Timestep', floc=20)
plot_distribution_fit(journey_time, 'gamma', bins=30, name='Journey Time', floc=0)


#plot_distribution_fit_manual(out_start_timestep, 'gamma', bins=70, name='Out Start Timestep', params=(20, 0, 1), floc=0)
plt.show()