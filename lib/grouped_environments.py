import atexit
import random
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from lib.tracking import RenderEpisodes


class GroupedEnvironments:
    """
    An environment grouping class of size `batch_size` that contains multiple environments that are reset with identical
    seeds within each group of size `group_size`. Valid masks are accumulated over the group.
    """

    def __init__(
        self,
        env_name: str,
        group_size: int,
        batch_size: int,
        enable_group_initialization: bool = True,
        seed: int | None = None,
        max_steps: int | None = None,
        render_path: Path | None = None,
    ):
        assert group_size > 0 and batch_size > 0, "Group size and batch size must be positive"
        assert batch_size % group_size == 0, "Batch size must be divisible by group size"

        dummy_env = gym.make(env_name, render_mode="rgb_array")
        self.num_observations = dummy_env.observation_space.shape[0]  # type: ignore
        self.num_actions = dummy_env.action_space.n  # type: ignore

        self.env_name = env_name
        self.group_size = group_size
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.enable_group_initialization = enable_group_initialization

        self.current_step = 0
        self.max_steps = max_steps

        self.envs = gym.make_vec(env_name, num_envs=batch_size, vectorization_mode="sync", render_mode="rgb_array")

        # Initialize renderer before reset, so it's available during the first reset
        self.render = RenderEpisodes(render_path, batch_size, dummy_env)

        # Register cleanup on exit to ensure video resources are released
        atexit.register(self.render.close)

        self.reset()

    def reset(self) -> torch.Tensor:
        """
        Resets the environments with identical seeds within each group.
        """
        self.terminated_masks: list[np.ndarray] = []
        self.truncated_masks: list[np.ndarray] = []
        self.current_terminated_mask = np.zeros(self.batch_size, dtype=np.bool_)
        self.current_truncated_mask = np.zeros(self.batch_size, dtype=np.bool_)
        self.current_step = 0

        seeding_group_size = self.group_size if self.enable_group_initialization else 1
        group_seeds = [self.rng.randint(0, 2**32 - 1) for _ in range(self.batch_size // seeding_group_size)]
        group_seeds = [group_seeds[i // seeding_group_size] for i in range(self.batch_size)]
        obs, infos = self.envs.reset(seed=group_seeds)  # type: ignore

        self.render.reset()

        return self._transform_observation(obs)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Steps the environments with the given actions and accumulates rewards.
        Returns the observations in torch format and a valid mask.

        Args:
            actions: Tensor of actions (batch_size, steps)

        Returns:
            observations: Tensor of observations (batch_size, steps)
            rewards: Tensor of rewards (batch_size, steps)
            valid_mask: Tensor of valid masks (batch_size, steps)
        """

        actions_np = actions.cpu().numpy()
        obs, rewards, termination_mask, truncation_mask, infos = self.envs.step(actions_np)

        self.terminated_masks.append(self.current_terminated_mask)
        self.truncated_masks.append(self.current_truncated_mask)
        self.current_terminated_mask = np.logical_or(self.current_terminated_mask, termination_mask)
        self.current_truncated_mask = np.logical_or(self.current_truncated_mask, truncation_mask)
        current_valid_mask = np.logical_not(np.logical_or(self.current_terminated_mask, self.current_truncated_mask))

        # Check if all environments are done
        done = not bool(current_valid_mask.any())

        render_valid_mask = np.logical_not(np.logical_or(self.terminated_masks[-1], self.truncated_masks[-1]))
        self.render.step(self.envs, rewards * render_valid_mask, render_valid_mask)  # Regular rendering

        if self.max_steps is not None and self.current_step >= self.max_steps:
            done = True
        if done:
            self.render.step(self.envs, rewards * current_valid_mask, current_valid_mask)  # Show that all environments are done
        self.current_step += 1

        return self._transform_observation(obs), self._transform_rewards(rewards, render_valid_mask), done

    def get_terminated_mask(self) -> torch.Tensor:
        # (num_environment_steps, batch_size) -> (batch_size, num_environment_steps)
        terminated_masks = np.array(self.terminated_masks).T
        return torch.from_numpy(terminated_masks).to(torch.bool)

    def get_truncated_mask(self) -> torch.Tensor:
        # (num_environment_steps, batch_size) -> (batch_size, num_environment_steps)
        truncated_masks = np.array(self.truncated_masks).T
        return torch.from_numpy(truncated_masks).to(torch.bool)

    def get_current_terminated_mask(self) -> torch.Tensor:
        return torch.from_numpy(self.current_terminated_mask).to(torch.bool)

    def get_current_truncated_mask(self) -> torch.Tensor:
        return torch.from_numpy(self.current_truncated_mask).to(torch.bool)

    def _transform_observation(self, obs: np.ndarray) -> torch.Tensor:
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

    def _transform_rewards(self, rewards: np.ndarray, valid_mask: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(rewards * valid_mask).to(torch.float32)
