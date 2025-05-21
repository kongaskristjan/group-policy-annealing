import torch

from lib.grouped_environments import GroupedEnvironments
from lib.model import get_model
from lib.sample import sample_batch_episode


def test_sample_batch_episode():
    """Test sampling actions from a model."""
    # Create a model for CartPole-v1 which has 4 observations and 2 actions
    group_size, batch_size = 2, 6
    envs = GroupedEnvironments("CartPole-v1", group_size, batch_size, seed=42)
    model = get_model(envs.num_observations, envs.num_actions, hidden=[8])

    # Sample actions
    observations, actions, rewards, terminated_mask, truncated_mask = sample_batch_episode(model, envs)

    # Check the shapes of the tensors
    steps = actions.shape[1]
    assert observations.shape == (batch_size, steps, envs.num_observations)
    assert actions.shape == (batch_size, steps)
    assert rewards.shape == (batch_size, steps)
    assert terminated_mask.shape == (batch_size, steps)
    assert truncated_mask.shape == (batch_size, steps)

    # Check that actions are valid
    assert torch.all((actions >= 0) & (actions < envs.num_actions))
