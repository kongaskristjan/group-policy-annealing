import torch


def anneal_value_function(
    policy: torch.nn.Module,
    value: torch.nn.Module,
    observations: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    temperature: float,
    clip_eps: float,
    discount_factor: float,
    optim_steps: int,
) -> list[float]:
    """
    Anneal the policy model and value function with a batch of episodes.

    Args:
        policy: The policy model to anneal
        value: The value function to anneal
        observations: Tensor of observations (batch_size, steps, num_observations)
        actions: Tensor of actions the policy model took (batch_size, steps)
        rewards: Tensor of rewards (batch_size, steps)
        valid_mask: Tensor of valid masks (batch_size, steps)
        optimizer: The optimizer to use for the annealing.
        temperature: The temperature to use for the annealing.
        clip_eps: The clipping epsilon for the policy.
        discount_factor: The discount factor for the value function.
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

    # Clipping (currently outdated and not used for simplicity)
    # with torch.no_grad():
    #    output = policy(torch.reshape(observations, (batch_size * steps, num_observations)))
    #    output = output.view(batch_size, steps, -1)
    #    log_probs = torch.log_softmax(output, dim=2)
    #
    # initial_log_probs = _compute_output_log_probs(log_probs, actions, valid_mask, group_size)
    # target_log_probs = _compute_target_log_probs(rewards, temperature, group_size)
    # centered_target_log_probs = target_log_probs - torch.mean(target_log_probs, dim=1, keepdim=True)
    # centered_initial_log_probs = initial_log_probs - torch.mean(initial_log_probs, dim=1, keepdim=True)
    # clipped_target_log_probs = torch.clamp(centered_target_log_probs, centered_initial_log_probs - clip_eps, centered_initial_log_probs + clip_eps)

    # Training loop
    losses = []
    for _ in range(optim_steps):
        optimizer.zero_grad()
        policy_output = policy(torch.reshape(observations, (batch_size * steps, num_observations)))
        value_output = value(torch.reshape(observations, (batch_size * steps, num_observations)))
        policy_output = policy_output.view(batch_size, steps, -1)
        value_output = value_output.view(batch_size, steps)

        current_loss = value_annealing_loss(policy_output, value_output, actions, valid_mask, rewards, temperature, discount_factor)
        current_loss.backward()
        optimizer.step()
        losses.append(current_loss.item())

    return losses


def value_annealing_loss(
    policy_output: torch.Tensor,
    values: torch.Tensor,
    actions: torch.Tensor,
    valid_mask: torch.Tensor,
    rewards: torch.Tensor,
    temperature: float,
    discount_factor: float,
) -> torch.Tensor:
    """
    Compute the loss for a batch of episodes for annealing.

    Equation that needs to be approximated:

    1) selected_log_probs[batch, step] * temperature = (discount_factor * value[batch, step+1] - value[batch, step]) + reward[batch, step]
    2) value[batch, step] = 0 if valid_mask[batch, step] == 0 (value is 0 after the episode ended)

    We assume that the valid_mask is 0 for the last step in any episode.

    Args:
        policy_output: The output of the policy model (batch_size, steps, num_actions)
        value_output: The output of the value model (batch_size, steps)
        actions: The actions (batch_size, steps)
        valid_mask: The valid mask (batch_size, steps)
        rewards: The rewards (batch_size, steps)
        temperature: The temperature for the Boltzmann distribution
        discount_factor: The discount factor for the value function

    Returns:
        The annealing loss, comparing the value output history to the policy probabilities and rewards.
    """
    batch_size, steps, num_actions = policy_output.shape

    # Compute output and target log-probabilities (batch_size, steps)
    log_probs = torch.log_softmax(policy_output, dim=2)
    selected_log_probs = torch.gather(log_probs, dim=2, index=actions.unsqueeze(2)).squeeze(2)

    # Apply rule (2): value is 0 after the episode ended
    values = values * valid_mask

    # Prepare V(s_{t+1})
    next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1)

    # Equation (1) can be rewritten as:
    # loss_term = (selected_log_probs * temperature) - ((discount_factor * next_values_padded - current_values) + rewards)
    # We want to minimize the squared discrepancy.

    # Left-hand side (LHS) of equation (1)
    lhs = selected_log_probs * temperature

    # Right-hand side (RHS) of equation (1)
    value_target_difference = (discount_factor * next_values) - values
    rhs = value_target_difference + rewards

    # Calculate the squared discrepancy for each step
    discrepancy = lhs - rhs
    squared_discrepancy = discrepancy.pow(2)

    # Mask the squared discrepancy: loss is 0 for invalid steps
    masked_squared_discrepancy = squared_discrepancy * valid_mask

    # The final loss is the mean of all losses per each (batch, step) value.
    # We should only average over the valid steps.
    sum_valid_mask = valid_mask.sum()
    mean_loss = masked_squared_discrepancy.sum() / (sum_valid_mask + 1e-6)

    return mean_loss
