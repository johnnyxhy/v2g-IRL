import torch
import torch.nn as nn
import numpy as np

# Observation normalization scales for the gap variant
# Order: [timestep, soc, soc_gap, energy_price, battery_capacity, time_to_next_journey, current_charger_power]
PROFIT_OBS_SCALES = np.array([96.0, 1.0, 1.0, 0.47, 2.0, 96.0, 22.0], dtype=np.float32)

class RewardNet(nn.Module):
    """
    Neural network that approximates the reward function R_θ(s, a).
    Takes normalized observation and action as input, outputs an unbounded scalar reward.

    Architecture: two hidden layers with tanh activations, linear output.
    Unbounded output gives PPO a strong learning signal; L2 regularization (AdamW
    weight_decay) prevents reward divergence.
    """

    def __init__(self, obs_dim=7, action_dim=1, hidden_dim=64, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            # Linear output — unbounded reward allows strong gradients for PPO.
            # L2 regularization (weight_decay in AdamW) prevents divergence.
        )

    def forward(self, obs, action):
        """
        Args:
            obs: (batch, obs_dim) normalized observations
            action: (batch, action_dim) or (batch,) actions
        Returns:
            (batch,) scalar rewards in [0, 1]
        """
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


def flatten_obs_dict(obs_dict, scales=PROFIT_OBS_SCALES):
    """
    Flatten a Dict observation (from DummyVecEnv or raw env) to a normalized 1D numpy array.

    Args:
        obs_dict: Dict with keys matching the V2G observation space
        scales: normalization divisors for each observation dimension
    
    Returns:
        np.ndarray of shape (obs_dim,) with normalized values
    """
    raw = np.array([
        float(obs_dict['timestep'].flatten()[0]),
        float(obs_dict['soc'].flatten()[0]),
        float(obs_dict['soc_gap'].flatten()[0]),
        float(obs_dict['energy_price'].flatten()[0]),
        float(obs_dict['battery_capacity'].flatten()[0]),
        float(obs_dict['time_to_next_journey'].flatten()[0]),
        float(obs_dict['current_charger_power'].flatten()[0]),
    ], dtype=np.float32)
    return raw / scales
