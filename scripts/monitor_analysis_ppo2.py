import pandas as pd
import numpy as np

df = pd.read_csv('models/DeepMaxEntIRL_PPO_profit_exp1_10/monitor.csv', skiprows=1)
df['cumsteps'] = df['l'].cumsum()
steps_per_epoch = 500_000

for ep in range(1, 6):
    mask = (df['cumsteps'] <= ep * steps_per_epoch) & (df['cumsteps'] > (ep-1) * steps_per_epoch)
    sub = df[mask]
    if len(sub) > 0:
        per_step = (sub['r'] / sub['l']).mean()
        print(f'Epoch {ep}: reward/step={per_step:.5f}, mean_len={sub["l"].mean():.1f}')
