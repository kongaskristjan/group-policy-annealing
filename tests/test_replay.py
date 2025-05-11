import torch

from lib.replay import ReplayBuffer


def test_replay_buffer_initialization():
    """Test the initialization of the replay buffer."""
    buffer_size, group_size = 10, 2
    buffer = ReplayBuffer(buffer_size, group_size)

    assert buffer.buffer_size == buffer_size
    assert buffer.group_size == group_size
    assert len(buffer.observations) == 0
    assert len(buffer.actions) == 0
    assert len(buffer.rewards) == 0
    assert len(buffer.valid_mask) == 0


def test_replay_buffer_add_and_sample():
    """Test adding to and sampling from the replay buffer."""
    buffer_size, group_size = 6, 2
    buffer = ReplayBuffer(buffer_size, group_size)

    # Create sample data
    batch_size = 4
    steps = 3
    num_observations = 2

    observations = torch.rand(batch_size, steps, num_observations)
    actions = torch.randint(0, 2, (batch_size, steps))
    rewards = torch.rand(batch_size, steps)  # Match dimensions with done_mask
    valid_mask = torch.zeros(batch_size, steps, dtype=torch.bool)
    # Set some episodes to be done at different timesteps
    valid_mask[0, 2] = False
    valid_mask[1, 1] = False
    valid_mask[2, 2] = False
    valid_mask[3, 1] = False

    # Add to buffer
    buffer.add(observations, actions, rewards, valid_mask)

    # Check buffer size
    assert len(buffer.observations) == batch_size
    assert len(buffer.actions) == batch_size
    assert len(buffer.rewards) == batch_size
    assert len(buffer.valid_mask) == batch_size

    # Sample from buffer
    sample_size = 4  # Must be divisible by group_size
    sampled_obs, sampled_actions, sampled_rewards, sampled_valid_mask = buffer.sample(sample_size)

    # Check shapes
    assert sampled_obs.shape[0] == sample_size
    assert sampled_actions.shape[0] == sample_size
    assert sampled_rewards.shape[0] == sample_size
    assert sampled_valid_mask.shape[0] == sample_size


def test_replay_buffer_overflow():
    """Test behavior when buffer is filled beyond capacity."""
    buffer_size, group_size = 4, 2
    buffer = ReplayBuffer(buffer_size, group_size)

    # Create data that will overflow the buffer
    batch_size = 6  # > buffer_size
    steps = 1
    observations = torch.rand(batch_size, steps, 2)
    actions = torch.randint(0, 2, (batch_size, steps))
    rewards = torch.rand(batch_size, steps)
    valid_mask = torch.zeros(batch_size, steps, dtype=torch.bool)  # All done, not valid

    # Add to buffer
    buffer.add(observations, actions, rewards, valid_mask)

    # Check buffer size
    assert len(buffer.observations) == buffer_size
    assert len(buffer.actions) == buffer_size
    assert len(buffer.rewards) == buffer_size
    assert len(buffer.valid_mask) == buffer_size
