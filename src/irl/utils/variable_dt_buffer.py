import numpy as np
import torch
from stable_baselines3.common.buffers import DictReplayBuffer
from sbx.common.type_aliases import ReplayBufferSamplesNp

class VariableDtReplayBuffer(DictReplayBuffer):
    """
    Based on SBX's DictReplayBuffer, but modified to store per-transition delta_t values

    Replay buffer that stores per-transition delta_t values and computes
    gamma^(delta_t) as the per-transition discount, passed directly to
    SBX's Bellman update via the discounts field of ReplayBufferSamplesNp.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device="auto",
        n_envs: int = 1,
        base_gamma: float = 0.99,
        **kwargs,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            **kwargs,
        )
        self.base_gamma = base_gamma
        self.delta_ts = np.ones((self.buffer_size,), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos):
        delta_t = infos[0].get('delta_t', 1) if infos else 1
        self.delta_ts[self.pos] = float(delta_t)
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env=None):
        upper_bound = self.buffer_size if self.full else self.pos
        indices = np.random.randint(0, upper_bound, size=batch_size)

        batch = self._get_samples(indices, env=env)

        # Compute gamma^(delta_t) as a torch tensor — SBX calls .numpy() on discounts
        adjusted_gammas = torch.tensor(
            self.base_gamma ** self.delta_ts[indices], dtype=torch.float32
        )

        return ReplayBufferSamplesNp(
            observations=batch.observations,
            actions=batch.actions,
            next_observations=batch.next_observations,
            dones=batch.dones,
            rewards=batch.rewards,
            discounts=adjusted_gammas,
        )