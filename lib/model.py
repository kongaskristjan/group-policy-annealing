import torch


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


def get_temperature(temp_start: float, temp_end: float, progress: float) -> float:
    """
    Exponential temperature annealing schedule.

    Args:
        temp_start: The initial temperature
        temp_end: The final temperature
        progress: The progress of the annealing (0 to 1)

    Returns:
        The temperature
    """
    return temp_start * (temp_end / temp_start) ** progress
