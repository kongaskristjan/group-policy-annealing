import torch

from lib.grouped_environments import GroupedEnvironments


def get_model(num_observations: int, num_actions: int, hidden: list[int]) -> torch.nn.Sequential:
    """
    Get a model with the given number of observations, actions, and hidden layers.
    """
    layer_sizes = [num_observations] + hidden + [num_actions]
    layers = []
    for i in range(len(layer_sizes) - 1):
        if i < len(layer_sizes) - 2:  # Not the last layer
            layers.append(
                torch.nn.Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                )
            )
            layers.append(torch.nn.ReLU())
        else:  # Last layer
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # Initialize with zeros
            layers[-1].weight.data.zero_()
            layers[-1].bias.data.zero_()
    return torch.nn.Sequential(*layers)


def sample_batch_episode(model: torch.nn.Module, envs: GroupedEnvironments) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample actions from the model in the given environment.
    Returns:
        - observations: Tensor of observations (batch_size, steps, num_observations)
        - actions: Tensor of sampled actions (batch_size, steps)
        - rewards: Tensor of rewards from the environment (batch_size,)
        - done_mask: Tensor of done masks (batch_size, steps)
    """
    # Reset the environment to start a new episode
    actions: list[torch.Tensor] = []
    observations: list[torch.Tensor] = []

    # Simulate the environment and policy as long as needed
    done = False
    cur_observations = envs.reset()  # (batch_size, num_observations)
    while not done:
        observations.append(cur_observations)

        # Sample an action from the model
        logits = model(cur_observations)  # (batch_size, num_actions)
        cur_probs = torch.nn.functional.softmax(logits, dim=1)  # (batch_size, num_actions)
        cur_actions = torch.multinomial(cur_probs, num_samples=1).squeeze(-1)  # (batch_size,)
        cur_observations, done = envs.step(cur_actions)  # (batch_size, num_observations), bool

        actions.append(cur_actions)

    # Get the rewards and done mask from the environment
    rewards = envs.get_rewards()  # (batch_size,)
    done_mask = envs.get_done_mask()  # (batch_size, steps)

    # Transpose the actions and probs (steps, batch_size) -> (batch_size, steps)
    observations_t = torch.transpose(torch.stack(observations), 0, 1)  # (batch_size, steps, num_observations)
    actions_t = torch.transpose(torch.stack(actions), 0, 1)  # (batch_size, steps)

    return observations_t, actions_t, rewards, done_mask
