import torch


def anneal_step(
    model: torch.nn.Module,
    actions: torch.Tensor,
    probs: torch.Tensor,
    rewards: torch.Tensor,
    done_mask: torch.Tensor,
    learning_rate: float,
    temperature: float,
    optim_steps: int,
) -> float:
    """
    Anneal the model for one step.
    """
    pass
