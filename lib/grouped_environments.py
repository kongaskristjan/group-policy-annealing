import random

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType


class GroupedEnvironments:
    """
    An environment grouping class of size `batch_size` that contains multiple environments that are reset with identical
    seeds within each group of size `group_size`. Rewards and dones are accumulated over the group.
    """

    def __init__(self, env_name: str, group_size: int, batch_size: int, seed: int | None = None):
        assert group_size > 0 and batch_size > 0, "Group size and batch size must be positive"
        assert batch_size % group_size == 0, "Batch size must be divisible by group size"

        info_env = gym.make(env_name)
        self.num_observations = info_env.observation_space.shape[0]
        self.num_actions = info_env.action_space.n

        self.env_name = env_name
        self.group_size = group_size
        self.batch_size = batch_size
        self.rng = random.Random(seed)

        self.envs = gym.make_vec(env_name, num_envs=batch_size, vectorization_mode="sync")
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Resets the rewards and environments with identical seeds within each group.
        """
        self.rewards = np.zeros((self.batch_size,), dtype=np.float32)
        self.done_masks: list[np.ndarray] = []

        group_seeds = [self.rng.randint(0, 2**32 - 1) for _ in range(self.batch_size // self.group_size)]
        group_seeds = [group_seeds[i // self.group_size] for i in range(self.batch_size)]
        obs, infos = self.envs.reset(seed=group_seeds)
        return self._transform_observation(obs)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Steps the environments with the given actions and accumulates rewards.
        """
        obs, rewards, done_mask, truncations, infos = self.envs.step(actions)
        if len(self.done_masks) > 0:
            done_mask = np.logical_or(done_mask, self.done_masks[-1])
        self.rewards += rewards * np.logical_not(done_mask)
        self.done_masks.append(done_mask)

        done = done_mask.all()
        return self._transform_observation(obs), done

    def get_rewards(self) -> np.ndarray:
        assert self.env_name == "CartPole-v1", "Only CartPole-v1 is supported for now"

        # Normalize rewards to be between 0 and 1 (CartPole-v1 max reward is 500)
        return self.rewards / 500

    def get_done_mask(self) -> np.ndarray:
        # (num_environment_steps, batch_size) -> (batch_size, num_environment_steps)
        return np.array(self.done_masks).T

    def _transform_observation(self, obs: ObsType) -> np.ndarray:
        """
        Returns the transformed observations for each environment in the batch.
        """
        assert self.env_name == "CartPole-v1", "Only CartPole-v1 is supported for now"

        obs[:, 0] /= 2.4  # cart_position (-2.4, 2.4) -> (-1, 1)
        obs[:, 1] /= 2.4 * 10  # cart_velocity (-inf, inf) -> (-inf, inf)
        obs[:, 2] /= 0.418  # pole_angle (-0.418, 0.418) -> (-1, 1)
        obs[:, 3] /= 0.418 * 10  # pole_velocity (-inf, inf) -> (-inf, inf)

        return obs
