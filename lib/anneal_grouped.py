import torch

from lib.loss_grouped import clip_target_log_probs, grouped_loss


def anneal_grouped(
    policy: torch.nn.Module,
    observations: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    terminated_mask: torch.Tensor,
    truncated_mask: torch.Tensor,
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
        terminated_mask: Tensor of terminated masks (batch_size, steps)
        truncated_mask: Tensor of truncated masks (batch_size, steps)
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
    terminated_mask = terminated_mask.to(device)
    truncated_mask = truncated_mask.to(device)
    valid_mask = torch.logical_not(torch.logical_or(terminated_mask, truncated_mask))

    rewards = torch.sum(rewards * valid_mask, dim=1)  # (batch_size,)
    clipped_target_log_probs = clip_target_log_probs(policy, observations, actions, rewards, valid_mask, temperature, group_size, clip_eps)

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
