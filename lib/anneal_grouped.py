import math

import torch


def anneal_grouped(
    policy: torch.nn.Module,
    observations: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    temperature: float,
    clip_eps: float,
    group_size: int,
    optim_steps: int,
) -> list[float]:
    """
    Anneal the policy model with a batch of episodes.

    Args:
        policy: The policy model to anneal
        observations: Tensor of observations (batch_size, steps, num_observations)
        actions: Tensor of actions the policy model took (batch_size, steps)
        rewards: Tensor of rewards (batch_size, steps)
        valid_mask: Tensor of valid masks (batch_size, steps)
        optimizer: The optimizer to use for the annealing.
        temperature: The temperature to use for the annealing.
        clip_eps: The clipping epsilon for the policy.
        group_size: The number of environments in each group with identical environment seeds
        optim_steps: The number of optimization steps to take.

    Returns:
        The loss during each step of the annealing.
    """
    device = next(policy.parameters()).device
    assert all(p.device == device for p in policy.parameters())  # Check that all parameters are on the same device

    batch_size, steps, num_observations = observations.shape
    observations = observations.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    valid_mask = valid_mask.to(device)

    rewards = torch.sum(rewards * valid_mask, dim=1)  # (batch_size,)

    with torch.no_grad():
        output = policy(torch.reshape(observations, (batch_size * steps, num_observations)))
        output = output.view(batch_size, steps, -1)
        log_probs = torch.log_softmax(output, dim=2)

    initial_log_probs = _compute_output_log_probs(log_probs, actions, valid_mask, group_size)
    target_log_probs = _compute_target_log_probs(rewards, temperature, group_size)
    centered_target_log_probs = target_log_probs - torch.mean(target_log_probs, dim=1, keepdim=True)
    centered_initial_log_probs = initial_log_probs - torch.mean(initial_log_probs, dim=1, keepdim=True)
    clipped_target_log_probs = torch.clamp(centered_target_log_probs, centered_initial_log_probs - clip_eps, centered_initial_log_probs + clip_eps)

    # Training loop
    losses = []
    for _ in range(optim_steps):
        optimizer.zero_grad()
        output = policy(torch.reshape(observations, (batch_size * steps, num_observations)))
        output = output.view(batch_size, steps, -1)
        log_probs = torch.log_softmax(output, dim=2)
        current_loss = grouped_loss(log_probs, actions, valid_mask, clipped_target_log_probs, group_size)
        current_loss.backward()
        optimizer.step()
        losses.append(current_loss.item())

    return losses


def grouped_loss(
    log_probs: torch.Tensor,
    actions: torch.Tensor,
    valid_mask: torch.Tensor,
    target_log_probs: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    Compute the loss for grouped batch of episodes for the annealing.

    Args:
        log_probs: The log probabilities of the actions (batch_size, steps, num_actions)
        actions: The actions (batch_size, steps)
        valid_mask: The valid mask (batch_size, steps)
        target_diffs: The target log-ratio matrix of the selected action's probabilities (num_groups, group_size, group_size)
        group_size: The number of environments in each group with identical environment seeds


    Returns:
        The annealing loss, comparing the probability of the actions taken to the target probability
        computed from the rewards and temperature by Boltzmann distribution.
    """
    batch_size, steps, num_actions = log_probs.shape
    num_groups = batch_size // group_size

    # Compute output and target log-probabilities (num_groups, group_size)
    output_log_probs = _compute_output_log_probs(log_probs, actions, valid_mask, group_size)

    # Boltzmann distribution only applies to probability ratios (or log probability differences),
    # thus we need to center the probability differences around 0 within each group
    log_prob_delta = output_log_probs - target_log_probs
    centered_log_prob_delta = log_prob_delta - torch.mean(log_prob_delta, dim=1, keepdim=True)

    # Compute the loss based on output and target differences
    loss = centered_log_prob_delta**2  # (num_groups, group_size)
    valid_values = num_groups * group_size
    loss = torch.sum(loss) / valid_values  # ()

    return loss


def _compute_output_log_probs(log_probs: torch.Tensor, actions: torch.Tensor, valid_mask: torch.Tensor, group_size: int) -> torch.Tensor:
    batch_size, steps, num_actions = log_probs.shape
    num_groups = batch_size // group_size

    # Group based view
    log_probs = log_probs.view(num_groups, group_size, steps, num_actions)
    actions = actions.view(num_groups, group_size, steps)
    valid_mask = valid_mask.view(num_groups, group_size, steps)

    # Gather the log-probabilities of the actions taken
    selected_log_probs = torch.gather(log_probs, dim=3, index=actions.unsqueeze(3)).squeeze(3)  # (num_groups, group_size, steps)

    # Assign the equivalent of random actions with equal probability to the actions after the episode is done
    selected_log_probs = selected_log_probs.masked_fill(torch.logical_not(valid_mask), math.log(1 / num_actions))

    # Compute the logarithm of total probability of the selected action's (computed as sum of log probabilities) (num_groups, group_size)
    sum_log_probs = torch.sum(selected_log_probs, dim=2)  # (num_groups, group_size)

    return sum_log_probs


def _compute_target_log_probs(rewards: torch.Tensor, temperature: float, group_size: int) -> torch.Tensor:
    # Group based view
    num_groups = rewards.shape[0] // group_size
    rewards = rewards.view(num_groups, group_size)

    # Compute the target log-ratios of the selected action's probabilities (num_groups, group_size)
    target_log_probs = rewards / temperature  # (num_groups, group_size)

    return target_log_probs
