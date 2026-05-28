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

journey_distance = extract_values(df, 'Upcoming_Trip_Distance_Miles', 0)
out_speed = extract_values(df, 'Average_Speed_Out_mph', 0)
return_speed = extract_values(df, 'Average_Speed_Return_mph', 0)
journey_speed = np.concatenate([out_speed, return_speed])

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
def plot_distribution_fit(data, distribution_name, bins=30, name='Data', floc=0, ax=None):

    params = fit_distribution(data, distribution_name, floc=floc)

    if ax is None:
        _, ax = plt.subplots()

    ax.hist(data, bins=bins, density=True, alpha=0.5, color='g', label='Data Histogram')

    data_min, data_max = np.min(data), np.max(data)
    x = np.linspace(data_min, data_max, 500)

    if distribution_name == 'norm':
        mu, sigma = params
        ax.plot(x, stats.norm.pdf(x, mu, sigma), linewidth=2, label=f"Normal (mu={mu:.4f}, sigma={sigma:.4f})")
        ax.set_title(f'Normal Distribution Fit for {name}')
    elif distribution_name == 'gamma':
        a, loc, scale = params
        ax.plot(x, stats.gamma.pdf(x, a, loc=loc, scale=scale), linewidth=2, label=f"Gamma (a={a:.4f}, loc={loc:.4f}, scale={scale:.4f})")
        ax.set_title(f'Gamma Distribution Fit for {name}')

    if name == "Initial SoC":
        ax.set_xlabel('SoC')
    elif name == "Out Start Timestep":
        ax.set_xlabel('Timestep')
    elif name == "Return Start Timestep":
        ax.set_xlabel('Timestep')
    elif name == "Journey Speed":
        ax.set_xlabel('Journey Speed')
    elif name == "Journey Distance":
        ax.set_xlabel('Journey Distance')

    ax.set_ylabel('Density')
    ax.legend()

    print(f"Fitted parameters for {name} ({distribution_name}): {params}")

# Manual Option
def plot_distribution_fit_manual(data, distribution_name, bins=30, name='Data', params=None, floc=0, ax=None):

    if ax is None:
        _, ax = plt.subplots()

    ax.hist(data, bins=bins, density=True, alpha=0.5, color='g', label='Data Histogram')

    data_min, data_max = np.min(data), np.max(data)
    x = np.linspace(data_min, data_max, 500)

    if distribution_name == 'norm':
        mu, sigma = params
        ax.plot(x, stats.norm.pdf(x, mu, sigma), linewidth=2, label=f"Normal (mu={mu:.4f}, sigma={sigma:.4f})")
    elif distribution_name == 'gamma':
        a, loc, scale = params
        ax.plot(x, stats.gamma.pdf(x, a, loc=loc, scale=scale), linewidth=2, label=f"Gamma (a={a:.4f}, loc={loc:.4f}, scale={scale:.4f})")

    ax.set_title(f'{distribution_name} Distribution Fit for {name}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()

# Exponential distribution plot
def plot_exponential_fit(data, bins=30, name='Data', ax=None):
    if ax is None:
        _, ax = plt.subplots()

    ax.hist(data, bins=bins, density=True, alpha=0.5, color='g', label='Data Histogram')

    data_min, data_max = np.min(data), np.max(data)
    x = np.linspace(data_min, data_max, 500)

    loc, scale = stats.expon.fit(data)
    ax.plot(x, stats.expon.pdf(x, loc=loc, scale=scale), linewidth=2, label=f"Exponential (loc={loc:.4f}, scale={scale:.4f})")

    ax.set_title(f'Exponential Distribution Fit for {name}')
    ax.set_xlabel('Journey Distance')
    ax.set_ylabel('Density')
    ax.legend()

    print(f"Fitted parameters for {name} (exponential): loc={loc}, scale={scale}")

# --- EXAMPLE USAGE ---
fig, axes = plt.subplots(2, 3, figsize=(18, 8))

plot_distribution_fit(soc_initial, 'norm', bins=50, name='Initial SoC', ax=axes[0, 0])
# plot_distribution_fit(energy_per_journey_minute, 'gamma', bins=50, name='Energy per Journey Minute', ax=axes[?])
plot_distribution_fit(out_start_timestep, 'gamma', bins=20, name='Out Start Timestep', ax=axes[0, 1])
plot_distribution_fit(return_start_timestep, 'gamma', bins=20, name='Return Start Timestep', floc=20, ax=axes[0, 2])
# plot_distribution_fit(journey_time, 'gamma', bins=30, name='Journey Time', floc=0, ax=axes[?])
plot_distribution_fit(journey_speed, 'gamma', bins=40, name='Journey Speed', ax=axes[1, 0])
plot_exponential_fit(journey_distance, bins=50, name='Journey Distance', ax=axes[1, 1])

axes[1, 2].set_visible(False)

fig.tight_layout()

#plot_distribution_fit_manual(out_start_timestep, 'gamma', bins=70, name='Out Start Timestep', params=(20, 0, 1), floc=0)
plt.show()