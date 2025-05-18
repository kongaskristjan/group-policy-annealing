from pathlib import Path

import torch

from lib.loss_value_function import value_annealing_loss
from lib.tracking import RenderValue


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
    render_path: Path | None = None,
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
    render: RenderValue | None = None
    if render_path is not None:
        render = RenderValue("Render value over annealing steps", render_path, observations[0], actions[0], rewards[0], valid_mask[0])
        render.update(policy, value, temperature, discount_factor)

    for _ in range(optim_steps):
        optimizer.zero_grad()
        policy_output = policy(torch.reshape(observations, (batch_size * steps, num_observations)))
        value_output = value(torch.reshape(observations, (batch_size * steps, num_observations)))
        policy_output = policy_output.view(batch_size, steps, -1)
        value_output = value_output.view(batch_size, steps)

        current_loss, debug_data = value_annealing_loss(policy_output, value_output, actions, valid_mask, rewards, temperature, discount_factor)
        current_loss.backward()
        optimizer.step()
        losses.append(current_loss.item())
        if render is not None:
            render.update(policy, value, temperature, discount_factor)

    if render is not None:
        render.close()
    return losses
