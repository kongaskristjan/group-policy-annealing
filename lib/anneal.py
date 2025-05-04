import math

import torch


def anneal_batch_episode(
    model: torch.nn.Module,
    observations: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    done_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    temperature: float,
    group_size: int,
    optim_steps: int,
) -> list[float]:
    """
    Anneal the model with a batch of episodes.

    Args:
        model: The model to anneal
        observations: Tensor of observations (batch_size, steps, num_observations)
        actions: Tensor of actions the model took (batch_size, steps)
        rewards: Tensor of rewards (batch_size,)
        done_mask: Tensor of done masks (batch_size, steps)
        optimizer: The optimizer to use for the annealing.
        temperature: The temperature to use for the annealing.
        group_size: The number of environments in each group with identical environment seeds
        optim_steps: The number of optimization steps to take.

    Returns:
        The loss during each step of the annealing.
    """
    device = next(model.parameters()).device
    assert all(p.device == device for p in model.parameters())  # Check that all parameters are on the same device

    batch_size, steps, num_observations = observations.shape
    observations = observations.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    done_mask = done_mask.to(device)

    # Training loop
    losses = []
    for _ in range(optim_steps):
        optimizer.zero_grad()
        output = model(torch.reshape(observations, (batch_size * steps, num_observations)))
        current_loss = annealing_loss(output.view(batch_size, steps, -1), actions, rewards, done_mask, temperature, group_size)
        current_loss.backward()
        optimizer.step()
        losses.append(current_loss.item())

    return losses


def annealing_loss(
    output: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    done_mask: torch.Tensor,
    temperature: float,
    group_size: int,
    apply_softmax: bool = True,
) -> torch.Tensor:
    """
    Compute the loss for the annealing.

    Args:
        output: The output of the model (batch_size, steps, num_actions)
        actions: The actions (batch_size, steps)
        rewards: The rewards (batch_size,)
        done_mask: The done mask (batch_size, steps)
        temperature: The temperature
        group_size: The number of environments in each group with identical environment seeds
        apply_softmax: Whether to apply softmax to the output

    Returns:
        The annealing loss, comparing the probability of the actions taken to the target probability
        computed from the rewards and temperature by Boltzmann distribution.
    """
    batch_size, steps, num_actions = output.shape
    num_groups = batch_size // group_size

    # Group based view
    output = output.view(num_groups, group_size, steps, num_actions)
    actions = actions.view(num_groups, group_size, steps)
    rewards = rewards.view(num_groups, group_size)
    done_mask = done_mask.view(num_groups, group_size, steps)

    # Gather the log-probabilities of the actions taken
    if apply_softmax:
        log_probs = torch.log_softmax(output, dim=3)  # (num_groups, group_size, steps, num_actions)
    else:
        log_probs = torch.log(output)  # (num_groups, group_size, steps, num_actions)
    selected_log_probs = torch.gather(log_probs, dim=3, index=actions.unsqueeze(3)).squeeze(3)  # (num_groups, group_size, steps)

    # Assign the equivalent of random actions with equal probability to the actions after the episode is done
    selected_log_probs = selected_log_probs.masked_fill(done_mask, math.log(1 / num_actions))

    # Compute the log-ratio matrix of the selected action's probabilities (num_groups, group_size, group_size)
    sum_log_probs = torch.sum(selected_log_probs, dim=2)  # (num_groups, group_size)
    log_prob_matrix = sum_log_probs.unsqueeze(1) - sum_log_probs.unsqueeze(2)  # (num_groups, group_size, group_size)

    # Target log-ratio matrix within each group based on rewards and temperature (num_groups, group_size, group_size)
    reward_diffs = rewards.unsqueeze(1) - rewards.unsqueeze(2)  # (num_groups, group_size, group_size)
    target_log_prob_matrix = reward_diffs / temperature  # (num_groups, group_size, group_size)

    # Compute the loss for each group
    loss = (log_prob_matrix - target_log_prob_matrix) ** 2  # (num_groups, group_size, group_size)
    loss = torch.sum(loss) / 2.0  # ()

    return loss


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
