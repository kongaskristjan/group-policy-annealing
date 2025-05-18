import torch


def value_annealing_loss(
    policy_output: torch.Tensor,
    values: torch.Tensor,
    actions: torch.Tensor,
    valid_mask: torch.Tensor,
    rewards: torch.Tensor,
    temperature: float,
    discount_factor: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
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
        The debug data contains the selected log probabilities, the values, the left-hand side and the right-hand side of the equation.
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

    return mean_loss, (selected_log_probs, values, lhs, rhs)
