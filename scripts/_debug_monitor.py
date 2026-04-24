import pandas as pd, numpy as np

df = pd.read_csv('models/DeepMaxEntIRL_discrete_exp1/monitor.csv', skiprows=1)
df.columns = ['r', 'l', 't']
steps_per_epoch = 300000
df['step'] = df['l'].cumsum()
df['epoch'] = (df['step'] // steps_per_epoch).astype(int) + 1

print("Episode reward stats per epoch:")
for epoch, grp in df.groupby('epoch'):
    n = len(grp)
    rmean = grp['r'].mean()
    rstd = grp['r'].std()
    lmean = grp['l'].mean()
    print(f"  Epoch {epoch:2d}: n={n:5d}, reward mean={rmean:7.3f}, std={rstd:6.3f}, ep_len mean={lmean:.1f}")

print()
print(f"Overall: {len(df)} episodes")
print(f"Episode length mean={df['l'].mean():.1f} (expert mean ~25 actions)")
print(f"Reward mean={df['r'].mean():.3f}, std={df['r'].std():.3f}")

# Check if episode length is changing (exploitation indicator)
print("\nEpisode length per epoch (did PPO exploit short/long episodes?):")
for epoch, grp in df.groupby('epoch'):
    lmean = grp['l'].mean()
    lmin = grp['l'].min()
    lmax = grp['l'].max()
    print(f"  Epoch {epoch:2d}: mean={lmean:.1f}, min={lmin}, max={lmax}")
