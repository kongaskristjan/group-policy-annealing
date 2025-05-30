import math

import torch
import torch.nn.functional as F


def value_gradient_annealing_loss(
    policy_output: torch.Tensor,
    value_output: torch.Tensor,
    actions: torch.Tensor,
    terminated_mask: torch.Tensor,
    truncated_mask: torch.Tensor,
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
    The valid_mask is calculated as: not (terminated_mask OR truncated_mask)

    Args:
        policy_output: The output of the policy model (batch_size, steps, num_actions) (float)
        value_output: The output of the value model (batch_size, steps) (float)
        actions: The actions (batch_size, steps) (int)
        terminated_mask: The terminated mask (batch_size, steps) (bool)
        truncated_mask: The truncated mask (batch_size, steps) (bool)
        rewards: The rewards (batch_size, steps) (float)
        temperature: The temperature for the Boltzmann distribution
        discount_factor: The discount factor for the value function

    Returns:
        The annealing loss, comparing the value output history to the policy probabilities and rewards.
        The debug data contains the selected log probabilities, the values, the left-hand side and the right-hand side of the equation.
    """

    valid_mask = torch.logical_not(torch.logical_or(terminated_mask, truncated_mask))
    batch_size, steps, num_actions = policy_output.shape

    # Compute output and target log-probabilities (batch_size, steps)
    log_probs = torch.log_softmax(policy_output, dim=2)
    selected_log_probs = torch.gather(log_probs, dim=2, index=actions.unsqueeze(2)).squeeze(2)

    # Apply rule (2): value is 0 after the episode ended
    value_output = value_output * valid_mask

    # Prepare V(s_{t+1})
    next_values = torch.cat([value_output[:, 1:], torch.zeros_like(value_output[:, :1])], dim=1)
    next_truncated_mask = torch.cat([truncated_mask[:, 1:], torch.zeros_like(truncated_mask[:, :1])], dim=1)

    # Equation (1) can be rewritten as:
    # loss_term = (selected_log_probs * temperature) - ((discount_factor * next_values_padded - current_values) + rewards)
    # We want to minimize the squared discrepancy.

    # Left-hand side (LHS) of equation (1)
    lhs = selected_log_probs * temperature

    # Right-hand side (RHS) of equation (1)
    value_target_difference = (discount_factor * next_values) - value_output
    rhs = value_target_difference + rewards

    # Calculate the squared discrepancy for each step
    discrepancy = (lhs - rhs) * torch.logical_not(next_truncated_mask)
    squared_discrepancy = discrepancy.pow(2)

    # Mask the squared discrepancy: loss is 0 for invalid steps
    masked_squared_discrepancy = squared_discrepancy * valid_mask

    # The final loss is the mean of all losses per each (batch, step) value.
    # We should only average over the valid steps.
    sum_valid_mask = valid_mask.sum()
    mean_loss = masked_squared_discrepancy.sum() / (sum_valid_mask + 1e-6)

    return mean_loss, (selected_log_probs, lhs, rhs, discrepancy)


def value_direct_annealing_loss(
    policy_output: torch.Tensor,
    value_output: torch.Tensor,
    actions: torch.Tensor,
    terminated_mask: torch.Tensor,
    truncated_mask: torch.Tensor,
    rewards: torch.Tensor,
    temperature: float,
    discount_factor: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Compute the loss for a batch of episodes for annealing.

    The equation to be approximated is based on the following equations:

    1) P_trajectory = K * exp(-E_trajectory / T)  (Boltzmann distribution for equilibrium thermodynamics, where K > 0 is arbitrary constant)
    2) P_trajectory = P1 * P2 * ... * Pn (probability of action sequence is product of probabilities of individual actions)
    3) E_trajectory = -R_trajectory = - (R1 + R2 + ... + Rn)  (energy is negative reward, and total reward is sum of rewards)

    Now, we can combine these equations:

    P1 * P2 * ... * Pn = K * exp((R1 + R2 + ... + Rn) / T)

    Taking the logarithm:

    log P1 + log P2 + ... + log Pn = log K + (R1 / T + R2 / T + ... + Rn / T)

    Rearrange the equation:

    -log K = (R1 / T - log P1) + (R2 / T - log P2) + ... + (Rn / T - log Pn)

    Now, keeping in mind that K is an arbitrary constant, we can instead define the left-hand side as the value of the state V = -log K:

    V = (R1 / T - log P1) + (R2 / T - log P2) + ... + (Rn / T - log Pn)

    Which is the basis of the loss function. However, in order to speed up training, we add a discount factor to the value function:

    V = (R1 / T - log P1) + discount_factor * (R2 / T - log P2) + ... + discount_factor ** (n-1) * (Rn / T - log Pn) (FINAL EQUATION)

    Which is the final equation we're trying to approximate.

    Termination is interpreted as getting 0 reward and having equal probability of all actions for the remaining steps.

    Truncation is interpreted as if the episode didn't end, and thus the last valid value is used for estimating the truncated rewards/log probabilities.

    Args:
        policy_output: The output of the policy model (batch_size, steps, num_actions) (float)
        value_output: The output of the value model (batch_size, steps) (float)
        actions: The actions (batch_size, steps) (int)
        terminated_mask: The terminated mask (batch_size, steps) (bool)
        truncated_mask: The truncated mask (batch_size, steps) (bool)
        rewards: The rewards (batch_size, steps) (float)
        temperature: The temperature for the Boltzmann distribution
        discount_factor: The discount factor for the value function

    Returns:
        The annealing loss, comparing the value output history to the policy probabilities and rewards.
        The debug data contains the selected log probabilities, the values, the left-hand side and the right-hand side of the equation.
    """
    batch_size, steps, num_actions = policy_output.shape

    # Compute the output log probabilities
    log_probs = torch.log_softmax(policy_output, dim=2)  # (batch_size, steps, num_actions)
    selected_log_probs = torch.gather(log_probs, dim=2, index=actions.unsqueeze(2)).squeeze(2)  # (batch_size, steps)

    # Filter by valid steps
    valid_mask = torch.logical_not(torch.logical_or(terminated_mask, truncated_mask))  # (batch_size, steps)
    selected_log_probs = valid_mask * selected_log_probs
    value_output = value_output * valid_mask  # (batch_size, steps)
    rewards = rewards * valid_mask  # (batch_size, steps)

    # Compute value target contribution R_i / T - log P_i for each step
    target_contrib = rewards / temperature - selected_log_probs  # (batch_size, steps)

    # Handle termination
    terminated_contrib = terminated_mask * torch.ones_like(selected_log_probs) * math.log(1 / num_actions)  # (batch_size, steps)
    target_contrib = target_contrib - terminated_contrib  # (batch_size, steps)

    # Handle truncation
    last_not_truncated = find_last_not_truncated(truncated_mask)  # (batch_size, steps)
    target_contrib = torch.logical_not(last_not_truncated) * target_contrib + last_not_truncated * value_output / (
        1 - discount_factor
    )  # (batch_size, steps)

    # Compute the discounted cumulative sum of the value contributions
    value_target = discount_cumsum(target_contrib, discount_factor, dim=1) * (1 - discount_factor)  # (batch_size, steps)

    # Compute the loss
    discrepancy = value_output - value_target  # (batch_size, steps)
    loss_sum = torch.sum(torch.square(discrepancy))
    loss_weight = torch.sum(valid_mask) + 1e-6
    loss = loss_sum / loss_weight

    return loss, (selected_log_probs, value_output, value_target, discrepancy)


def discount_cumsum(x: torch.Tensor, discount_factor: float, dim: int) -> torch.Tensor:
    weights = discount_factor ** torch.arange(x.size(dim), dtype=torch.float64, device=x.device)
    weights = weights.view(*[1] * dim + [x.size(dim)] + [1] * (x.dim() - dim - 1))
    x_weighted = weights * x
    weighted_sum = torch.flip(torch.cumsum(torch.flip(x_weighted, [dim]), dim=dim), [dim])  # Reversed cumulative sum
    return weighted_sum / weights


def find_last_not_truncated(truncated_mask: torch.Tensor) -> torch.Tensor:
    """
    Find the last non-truncated step in the truncated mask.
    """
    assert truncated_mask.ndim == 2, "Truncated mask must be 2D"

    # We know that steps are only truncated from the end, thus we can find the index of the last non-truncated step with the sum of the mask.
    sum_mask = torch.sum(truncated_mask, dim=1)  # (batch_size)
    last_truncated = truncated_mask.shape[1] - sum_mask  # (batch_size)
    last_not_truncated = last_truncated - 1  # (batch_size)
    one_hot_mask = F.one_hot(last_not_truncated, num_classes=truncated_mask.shape[1]).bool()  # (batch_size, steps)
    one_hot_mask[:, -1] = False
    return one_hot_mask
