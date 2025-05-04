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
