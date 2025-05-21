import torch

from lib.grouped_environments import GroupedEnvironments


def test_observation_seeding():
    env = GroupedEnvironments(env_name="CartPole-v1", group_size=2, batch_size=8)
    obs = env.reset()
    assert env.envs.num_envs == 8

    # Verify that environments within each group are identical but different from other groups
    assert torch.allclose(obs[0], obs[1])
    assert not torch.allclose(obs[1], obs[2])
    assert torch.allclose(obs[2], obs[3])
    assert not torch.allclose(obs[3], obs[4])
    assert torch.allclose(obs[4], obs[5])
    assert not torch.allclose(obs[5], obs[6])
    assert torch.allclose(obs[6], obs[7])
    assert not torch.allclose(obs[7], obs[0])


def test_valid_and_reward_accumulation():
    batch_size = 4

    # Test that done is False when the first step is taken
    env = GroupedEnvironments(env_name="CartPole-v1", group_size=1, batch_size=batch_size)
    input_actions = torch.zeros(batch_size, dtype=torch.int64)
    obs, rewards, done = env.step(input_actions)
    assert not done

    # Test that done is True when the last step is taken
    steps = 500
    sum_rewards = torch.zeros(batch_size)
    for _ in range(steps + 1):
        obs, rewards, done = env.step(input_actions)
        sum_rewards += rewards
    assert done

    # Test that terminated and truncated masks are initialized correctly
    terminated_mask = env.get_terminated_mask()
    truncated_mask = env.get_truncated_mask()

    # Test that all environments are valid at the first step
    assert not terminated_mask[:, 0].any()
    assert not truncated_mask[:, 0].any()

    # Test that all environments are either terminated or truncated at the last step
    assert torch.logical_or(terminated_mask[:, steps], truncated_mask[:, steps]).all()

    # Test that rewards are accumulated
    assert sum_rewards.shape == (batch_size,)

    # Rewards should be between 2 and 20
    assert (2 < sum_rewards).all()
    assert (sum_rewards < 20).all()
