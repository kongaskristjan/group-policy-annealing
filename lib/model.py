import torch

from lib.grouped_environments import GroupedEnvironments


def get_model(num_observations: int, num_actions: int, hidden: list[int]) -> torch.nn.Sequential:
    """
    Get a model with the given number of observations, actions, and hidden layers.
    """
    layer_sizes = [num_observations] + hidden + [num_actions]
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def sample_batch_episode(model: torch.nn.Module, envs: GroupedEnvironments) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample actions from the model in the given environment.
    Returns:
        - actions: Tensor of sampled actions (batch_size, steps)
        - probs: Tensor of selected action probabilities (batch_size, steps)
        - rewards: Tensor of rewards from the environment (batch_size,)
        - done_mask: Tensor of done masks (batch_size, steps)
    """
    # Reset the environment to start a new episode
    observations = envs.reset()
    probs: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []

    # Simulate the environment and policy as long as needed
    done = False
    while not done:
        logits = model(observations)
        current_probs = torch.nn.functional.softmax(logits, dim=1)
        current_actions = torch.multinomial(current_probs, num_samples=1).squeeze(-1)
        observations, done = envs.step(current_actions)
        selected_probs = current_probs[torch.arange(current_probs.shape[0]), current_actions]

        probs.append(selected_probs)
        actions.append(current_actions)

    # Get the rewards and done mask from the environment
    rewards = envs.get_rewards()
    done_mask = envs.get_done_mask()

    # Transpose the actions and probs to be (batch_size, num_steps)
    actions_t = torch.transpose(torch.stack(actions), 0, 1)
    probs_t = torch.transpose(torch.stack(probs), 0, 1)

    return actions_t, probs_t, rewards, done_mask
