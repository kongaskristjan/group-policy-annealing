import numpy as np

from lib.grouped_environments import GroupedEnvironments


def test_observation_seeding():
    env = GroupedEnvironments(env_name="CartPole-v1", group_size=2, batch_size=8)
    obs = env.reset()
    assert env.envs.num_envs == 8

    # Verify that environments within each group are identical but different from other groups
    assert np.allclose(obs[0], obs[1])
    assert not np.allclose(obs[1], obs[2])
    assert np.allclose(obs[2], obs[3])
    assert not np.allclose(obs[3], obs[4])
    assert np.allclose(obs[4], obs[5])
    assert not np.allclose(obs[5], obs[6])
    assert np.allclose(obs[6], obs[7])
    assert not np.allclose(obs[7], obs[0])


def test_done_and_reward_accumulation():
    batch_size = 4

    # Test that done is False when the first step is taken
    env = GroupedEnvironments(env_name="CartPole-v1", group_size=1, batch_size=batch_size)
    input_actions = np.array([0] * batch_size)
    obs, done = env.step(input_actions)
    assert not done

    # Test that done is True when the last step is taken
    steps = 500
    for _ in range(steps + 1):
        obs, done = env.step(input_actions)
    assert done

    # Test that done mask is False for all environments at the first step and True for all environments at the last step
    done_mask = env.get_done_mask()
    assert np.logical_not(done_mask[:, 0]).all()
    assert done_mask[:, steps].all()

    # Test that rewards are accumulated
    rewards = env.get_rewards()
    assert rewards.shape == (batch_size,)

    # Rewards should be between 2 and 20 (but they're normalized to be between 0 and 1)
    assert ((2 / steps) < rewards).all()
    assert (rewards < (20 / steps)).all()
