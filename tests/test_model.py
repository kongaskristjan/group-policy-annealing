import torch

from lib.model import get_model


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
