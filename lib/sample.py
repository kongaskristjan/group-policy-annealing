import torch

from lib.batch_environment import BatchEnvironment


def sample_batch_episode(
    policy: torch.nn.Module, envs: BatchEnvironment, validate: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample actions from the policy model in the given environment.
    Returns:
        - observations: Tensor of observations (batch_size, steps, num_observations)
        - actions: Tensor of sampled actions (batch_size, steps)
        - rewards: Tensor of rewards from the environment (batch_size, steps)
        - terminated_mask: Tensor of terminated masks (batch_size, steps)
        - truncated_mask: Tensor of truncated masks (batch_size, steps)
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
            logits = policy(cur_observations)  # (batch_size, num_actions)
            cur_probs = torch.nn.functional.softmax(logits, dim=1)  # (batch_size, num_actions)
            if validate:
                cur_actions = torch.argmax(cur_probs, dim=1)  # (batch_size,)
            else:
                cur_actions = torch.multinomial(cur_probs, num_samples=1).squeeze(-1)  # (batch_size,)
            cur_observations, cur_rewards, done = envs.step(cur_actions)  # (batch_size, num_observations), (batch_size), bool

            actions.append(cur_actions)
            rewards.append(cur_rewards)

        # Get the terminated and truncated masks from the environment
        terminated_mask = envs.get_terminated_mask()  # (batch_size, steps)
        truncated_mask = envs.get_truncated_mask()  # (batch_size, steps)
        current_terminated_mask = envs.get_current_terminated_mask()  # (batch_size,)
        current_truncated_mask = envs.get_current_truncated_mask()  # (batch_size,)

        # Transpose the rewards, actions and probs (steps, batch_size) -> (batch_size, steps)
        observations_t = torch.stack(observations, dim=1)  # (batch_size, steps, num_observations)
        actions_t = torch.stack(actions, dim=1)  # (batch_size, steps)
        rewards_t = torch.stack(rewards, dim=1)  # (batch_size, steps)

        # Add a dummy step at the end to account for the episode ending
        observations_t = torch.cat([observations_t, torch.zeros_like(observations_t[:, :1])], dim=1)
        actions_t = torch.cat([actions_t, torch.zeros_like(actions_t[:, :1])], dim=1)
        rewards_t = torch.cat([rewards_t, torch.zeros_like(rewards_t[:, :1])], dim=1)
        terminated_mask = torch.cat([terminated_mask, current_terminated_mask.unsqueeze(1)], dim=1)
        truncated_mask = torch.cat([truncated_mask, current_truncated_mask.unsqueeze(1)], dim=1)

    return observations_t, actions_t, rewards_t, terminated_mask, truncated_mask
