import torch


def anneal_batch_episode(
    model: torch.nn.Module,
    observations: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    done_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    temperature: float,
    optim_steps: int,
) -> list[float]:
    """
    Anneal the model with a batch of episodes.

    Args:
        model: The model to anneal
        observations: Tensor of observations (batch_size, steps)
        actions: Tensor of actions the model took (batch_size, steps)
        rewards: Tensor of rewards (batch_size,)
        done_mask: Tensor of done masks (batch_size, steps)
        optimizer: The optimizer to use for the annealing.
        temperature: The temperature to use for the annealing.
        optim_steps: The number of optimization steps to take.

    Returns:
        The loss during each step of the annealing.
    """

    observations = observations.to(model.device)
    actions = actions.to(model.device)
    rewards = rewards.to(model.device)
    done_mask = done_mask.to(model.device)

    # Training loop
    losses = []
    for _ in range(optim_steps):
        optimizer.zero_grad()
        output = model(observations)
        current_loss = loss(output, actions, rewards, done_mask, temperature)
        current_loss.backward()
        optimizer.step()
        losses.append(current_loss.item())

    return losses


def loss(output: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, done_mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    pass
