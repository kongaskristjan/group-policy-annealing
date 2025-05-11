import random
import warnings

import gymnasium as gym
import numpy as np
import torch
from gymnasium.core import ObsType


class GroupedEnvironments:
    """
    An environment grouping class of size `batch_size` that contains multiple environments that are reset with identical
    seeds within each group of size `group_size`. Dones are accumulated over the group.
    """

    def __init__(
        self,
        env_name: str,
        group_size: int,
        batch_size: int,
        enable_group_initialization: bool = True,
        seed: int | None = None,
        max_steps: int | None = None,
        render: bool = False,
    ):
        assert group_size > 0 and batch_size > 0, "Group size and batch size must be positive"
        assert batch_size % group_size == 0, "Batch size must be divisible by group size"

        info_env = gym.make(env_name)
        self.num_observations = info_env.observation_space.shape[0]
        self.num_actions = info_env.action_space.n

        self.env_name = env_name
        self.group_size = group_size
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.enable_group_initialization = enable_group_initialization

        self.current_step = 0
        self.max_steps = max_steps
        self.render = render

        self.envs = gym.make_vec(env_name, num_envs=batch_size, vectorization_mode="sync", render_mode="human" if render else None)
        self.reset()

    def reset(self) -> torch.Tensor:
        """
        Resets the environments with identical seeds within each group.
        """
        self.done_masks: list[np.ndarray] = []
        self.current_step = 0

        seeding_group_size = self.group_size if self.enable_group_initialization else 1
        group_seeds = [self.rng.randint(0, 2**32 - 1) for _ in range(self.batch_size // seeding_group_size)]
        group_seeds = [group_seeds[i // seeding_group_size] for i in range(self.batch_size)]
        obs, infos = self.envs.reset(seed=group_seeds)

        return self._transform_observation(obs)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Steps the environments with the given actions and accumulates rewards.
        Returns the observations in torch format and a done mask.

        Args:
            actions: Tensor of actions (batch_size, steps)

        Returns:
            observations: Tensor of observations (batch_size, steps)
            rewards: Tensor of rewards (batch_size, steps)
            done_mask: Tensor of done masks (batch_size, steps)
        """

        actions_np = actions.cpu().numpy()
        obs, rewards, termination_mask, truncation_mask, infos = self.envs.step(actions_np)

        done_mask = np.logical_or(termination_mask, truncation_mask)

        if self.render:
            self.envs.render()
        if len(self.done_masks) > 0:
            done_mask = np.logical_or(done_mask, self.done_masks[-1])
        self.done_masks.append(done_mask)

        done = bool(done_mask.all())
        if self.max_steps is not None and self.current_step >= self.max_steps:
            done = True
        self.current_step += 1
        return self._transform_observation(obs), self._transform_rewards(rewards, done_mask), done

    def get_done_mask(self) -> torch.Tensor:
        # (num_environment_steps, batch_size) -> (batch_size, num_environment_steps)
        done_masks = np.array(self.done_masks).T
        return torch.from_numpy(done_masks).to(torch.bool)

    def _transform_observation(self, obs: ObsType) -> torch.Tensor:
        """
        Returns the transformed observations for each environment in the batch.
        """
        if self.env_name == "CartPole-v1":
            obs[:, 0] /= 2.4  # cart_position (-2.4, 2.4) -> (-1, 1)
            obs[:, 1] /= 2.4 * 10  # cart_velocity (-inf, inf) -> (-inf, inf)
            obs[:, 2] /= 0.418  # pole_angle (-0.418, 0.418) -> (-1, 1)
            obs[:, 3] /= 0.418 * 10  # pole_velocity (-inf, inf) -> (-inf, inf)
        else:
            warnings.warn(
                f"Normalization coefficients not available for {self.env_name} observation space. You may still experiment with the environment."
            )

        return torch.from_numpy(obs).to(torch.float32)

    def _transform_rewards(self, rewards: np.ndarray, done_mask: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(rewards * np.logical_not(done_mask)).to(torch.float32)
