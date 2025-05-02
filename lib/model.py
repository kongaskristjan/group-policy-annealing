import torch

from lib.grouped_environments import GroupedEnvironments


def get_model(num_observations: int, num_actions: int, hidden: list[int]) -> torch.nn.Module:
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


def sample_actions(model: torch.nn.Module, envs: GroupedEnvironments) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample actions from the model in the given environment.
    """
    pass
