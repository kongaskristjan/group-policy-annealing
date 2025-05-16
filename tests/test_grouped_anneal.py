import torch

from lib.grouped_anneal import anneal_group_batch_episode, annealing_group_batch_loss, compute_target_log_probs
from lib.model import get_model


def test_anneal_batch_episode_simple():
    # Create a simple test with 2 trajectories of length 1
    # One trajectory has reward 1.0, the other has reward 0.0
    # Set up a linear model with no hidden layers - note that the layer is initialized with zeros
    policy = get_model(num_observations=1, num_actions=2, hidden=[])

    # Create observations, actions, rewards for 2 trajectories
    observations = torch.Tensor([[[-0.5]], [[0.5]]])  # 2 trajectories, 1 step, 1 observation
    actions = torch.zeros(2, 1, dtype=torch.long)  # Both take action 0
    rewards = torch.tensor([[1.0], [0.0]])  # First trajectory gets reward 1.0, second gets 0.0
    valid_mask = torch.tensor([[True], [True]])  # Both trajectories are valid

    # Setup optimizer with higher learning rate
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)

    # Run annealing with lower temperature and more steps
    anneal_group_batch_episode(
        policy=policy,
        observations=observations,
        actions=actions,
        rewards=rewards,
        valid_mask=valid_mask,
        optimizer=optimizer,
        temperature=1.0,  # Lower temperature makes the reward difference more significant
        clip_eps=100.0,  # No clipping
        group_size=2,  # Trajectories are compared with each other and form 2 groups
        optim_steps=20,  # Lots of optimization to ensure convergence
    )

    # Get probabilities after annealing
    with torch.no_grad():
        logits = policy(observations.view(2, 1))
        probs = torch.softmax(logits, dim=1)

        # Probability of action 0 for trajectory with reward 1.0
        p_higher_reward = probs[0, 0].item()
        # Probability of action 0 for trajectory with reward 0.0
        p_lower_reward = probs[1, 0].item()

    # Verify that annealing skewed the probabilities correctly
    # Difference between probabilities should be exp(1.0/1.0) = e with ideal annealing
    assert p_lower_reward < 0.5 * p_higher_reward and p_lower_reward > 0.25 * p_higher_reward
    # Note that both probabilities are lower than initially due to softmax gradient dynamics


def test_gradient_direction_and_symmetry():
    # Check gradient symmetry with reversed reward groups (2 groups of 2 trajectories)
    batch_size, group_size, num_actions = 4, 2, 2
    output = torch.zeros(batch_size, 1, num_actions, requires_grad=True)
    actions = torch.tensor([[0], [0], [0], [0]], dtype=torch.long)
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])  # Rewards: Group 1: [1, 0], Group 2: [0, 1] (reversed)
    valid_mask = torch.tensor([[True], [True], [True], [True]])

    target_log_probs = compute_target_log_probs(rewards, 1.0, group_size)
    log_probs = torch.log_softmax(output, dim=2)
    loss = annealing_group_batch_loss(log_probs, actions, valid_mask, target_log_probs, group_size)
    loss.backward()

    assert output.grad is not None

    # Group 1: traj 0 (high R) grad negative, traj 1 (low R) grad positive
    assert output.grad[0, 0, 0] < 0
    assert output.grad[1, 0, 0] > 0

    # Group 2: traj 2 (low R) grad positive, traj 3 (high R) grad negative
    assert output.grad[2, 0, 0] > 0
    assert output.grad[3, 0, 0] < 0

    # Check symmetry magnitude (approx due to calculation, might not be exact depending on implementation)
    assert torch.isclose(output.grad[0, 0, 0], -output.grad[1, 0, 0])
    assert torch.isclose(output.grad[2, 0, 0], -output.grad[3, 0, 0])
    assert torch.isclose(output.grad[0, 0, 0], output.grad[3, 0, 0])


def test_masking_and_unpicked_gradients():
    torch.manual_seed(42)

    # Case 3: Ensure masked steps and unpicked actions get no gradient
    group_size, num_actions, steps = 2, 2, 2
    output = torch.randn(group_size, steps, num_actions, requires_grad=True)
    softmax_output = torch.softmax(output, dim=2)
    actions = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # Traj 0: a0, a1; Traj 1: a1, a0
    rewards = torch.tensor([1.0, 0.0])
    valid_mask = torch.tensor([[True, False], [True, True]])  # Valid Mask: Mask out step 1 of trajectory 0

    target_log_probs = compute_target_log_probs(rewards, 1.0, group_size)
    log_probs = torch.log(softmax_output)
    loss = annealing_group_batch_loss(log_probs, actions, valid_mask, target_log_probs, group_size)
    softmax_output.retain_grad()
    loss.backward()

    assert softmax_output.grad is not None

    # Traj 0, Step 0 (Unmasked): Action 0 taken, Action 1 unpicked
    assert not torch.allclose(softmax_output.grad[0, 0, 0], torch.tensor(0.0))  # Grad for taken action
    assert torch.allclose(softmax_output.grad[0, 0, 1], torch.tensor(0.0))  # No grad for unpicked action

    # Traj 0, Step 1 (Masked): Both action grads should be zero
    assert torch.allclose(softmax_output.grad[0, 1, 0], torch.tensor(0.0))
    assert torch.allclose(softmax_output.grad[0, 1, 1], torch.tensor(0.0))


def test_zero_loss_for_perfect_match():
    # Loss should be zero if output matches Boltzmann dist (T=10)
    # Construct a temperature and reward that should result in a probability ratio of e
    # Reward difference = 3.0 and temperature = 3.0, thus prob ratio is exp(reward_diff/temp) = exp(3/3) = e
    group_size, num_actions, steps = 2, 2, 1
    temperature = 3.0
    rewards = torch.tensor([13.0, 10.0])  # Reward difference: 3.0
    actions = torch.tensor([[0], [0]], dtype=torch.long)
    valid_mask = torch.tensor([[True], [True]])

    # Construct output logits such that logP(a0|traj0) - logP(a0|traj1) = 1.0
    # Let logP(a0|traj0) = 0. Requires softmax(logits0)[0] = 1. Use large logit diff.
    # Let logP(a0|traj1) = -1. Requires softmax(logits1)[0] = exp(-1).
    # Logit L1_0 = 0, L1_1 = log(exp(1)-1) achieves this (as derived in thought process)
    output = torch.tensor(
        [
            [[10.0, -10.0]],  # Probability of this action is almost 1, or approx logP(a0)=0.
            [[0.0, torch.log(torch.exp(torch.tensor(1.0)) - 1)]],  # Approx logP(a0)=-1
        ],
        requires_grad=True,
    )

    target_log_probs = compute_target_log_probs(rewards, temperature, group_size)
    log_probs = torch.log_softmax(output, dim=2)
    loss = annealing_group_batch_loss(log_probs, actions, valid_mask, target_log_probs, group_size)

    # Loss should be close to zero
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
