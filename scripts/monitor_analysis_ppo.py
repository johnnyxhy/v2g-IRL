import pandas as pd
import numpy as np

df = pd.read_csv('models/DeepMaxEntIRL_PPO_profit_exp1_10/monitor.csv', skiprows=1)
print(f'Total episodes: {len(df)}')
print(f'Total steps (cumulative): {df["l"].sum()}')
print()
print('Episode length stats:')
print(df['l'].describe())
print()

steps_per_epoch = 500_000
df['cumsteps'] = df['l'].cumsum()
print('Per epoch (mean_r, reward/step, mean_len):')
for ep in range(1, 12):
    mask = (df['cumsteps'] <= ep * steps_per_epoch) & (df['cumsteps'] > (ep-1) * steps_per_epoch)
    sub = df[mask]
    if len(sub) > 0:
        per_step = (sub['r'] / sub['l']).mean()
        print(f'  Epoch {ep:2d}: mean_r={sub["r"].mean():.4f}, r/step={per_step:.5f}, mean_l={sub["l"].mean():.1f}')

print()
# Within-epoch trend for epoch 1
epoch1 = df[df['cumsteps'] <= steps_per_epoch]
n_win = 10
win_size = max(1, len(epoch1) // n_win)
print(f'Epoch 1 rolling (first->last):')
for i in range(n_win):
    w = epoch1.iloc[i*win_size:(i+1)*win_size]
    print(f'  Window {i+1}: mean_r={w["r"].mean():.4f}, mean_l={w["l"].mean():.1f}')
