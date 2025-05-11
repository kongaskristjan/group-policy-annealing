import torch

from lib.grouped_environments import GroupedEnvironments


def sample_batch_episode(model: torch.nn.Module, envs: GroupedEnvironments) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample actions from the model in the given environment.
    Returns:
        - observations: Tensor of observations (batch_size, steps, num_observations)
        - actions: Tensor of sampled actions (batch_size, steps)
        - rewards: Tensor of rewards from the environment (batch_size, steps)
        - done_mask: Tensor of done masks (batch_size, steps)
    """
    with torch.no_grad():
        # Reset the environment to start a new episode
        actions: list[torch.Tensor] = []
        observations: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []

        # Simulate the environment and policy as long as needed
        done = False
        cur_observations = envs.reset()  # (batch_size, num_observations)
        while not done:
            observations.append(cur_observations)

            # Sample an action from the model
            logits = model(cur_observations)  # (batch_size, num_actions)
            cur_probs = torch.nn.functional.softmax(logits, dim=1)  # (batch_size, num_actions)
            cur_actions = torch.multinomial(cur_probs, num_samples=1).squeeze(-1)  # (batch_size,)
            cur_observations, cur_rewards, done = envs.step(cur_actions)  # (batch_size, num_observations), (batch_size,), bool

            actions.append(cur_actions)
            rewards.append(cur_rewards)

        # Get the done mask from the environment
        done_mask = envs.get_done_mask()  # (batch_size, steps)

        # Transpose the rewards, actions and probs (steps, batch_size) -> (batch_size, steps)
        observations_t = torch.transpose(torch.stack(observations), 0, 1)  # (batch_size, steps, num_observations)
        actions_t = torch.transpose(torch.stack(actions), 0, 1)  # (batch_size, steps)
        rewards_t = torch.transpose(torch.stack(rewards), 0, 1)  # (batch_size, steps)

    return observations_t, actions_t, rewards_t, done_mask


def validate(model: torch.nn.Module, env_name: str, val_batch: int) -> float:
    """Run validation episodes and return the mean reward"""
    val_envs = GroupedEnvironments(env_name, 1, val_batch)
    observations, actions, rewards, done_mask = sample_batch_episode(model, val_envs)
    return torch.mean(rewards).item()


def render_episode(model: torch.nn.Module, env_name: str) -> None:
    """Render a single episode of the model in the given environment"""
    val_envs = GroupedEnvironments(env_name, 1, 1, render=True)
    observations, actions, rewards, done_mask = sample_batch_episode(model, val_envs)
