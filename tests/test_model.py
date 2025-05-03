import torch

from lib.grouped_environments import GroupedEnvironments
from lib.model import get_model, sample_batch_episode


def test_get_model_structure():
    """Test that the model has the correct structure."""
    # Create model
    num_observations, num_actions, hidden = 4, 2, [8]
    model = get_model(num_observations, num_actions, hidden)

    # Check the structure of the layers: (Input -> ReLU -> Output layer)
    assert len(model) == 3
    assert isinstance(model[0], torch.nn.Linear) and model[0].in_features == num_observations and model[0].out_features == hidden[0]
    assert isinstance(model[1], torch.nn.ReLU)
    assert isinstance(model[2], torch.nn.Linear) and model[2].in_features == hidden[0] and model[2].out_features == num_actions

    # Forward pass
    batch_size = 3
    observations = torch.rand(batch_size, num_observations)
    logits = model(observations)
    assert logits.shape == (batch_size, num_actions)


def test_get_model_different_hidden_sizes():
    """Test that models with different hidden layer configurations are created correctly."""
    assert len(get_model(4, 2, hidden=[])) == 1
    assert len(get_model(4, 2, hidden=[10])) == 3
    assert len(get_model(4, 2, hidden=[10, 20, 30])) == 7


def test_sample_batch_episode():
    """Test sampling actions from a model."""
    # Create a model for CartPole-v1 which has 4 observations and 2 actions
    group_size, batch_size = 2, 4
    envs = GroupedEnvironments("CartPole-v1", group_size, batch_size, seed=42)
    model = get_model(envs.num_observations, envs.num_actions, hidden=[8])

    # Sample actions
    actions, probs, rewards, done_mask = sample_batch_episode(model, envs)

    # Check the shapes of the tensors
    steps = actions.shape[1]
    assert actions.shape == (batch_size, steps)
    assert probs.shape == (batch_size, steps)
    assert rewards.shape == (batch_size,)
    assert done_mask.shape == (batch_size, steps)

    # Check that probs and actions are valid
    assert torch.all(0 < probs) and torch.all(probs < 1)
    assert torch.all((actions >= 0) & (actions < envs.num_actions))
