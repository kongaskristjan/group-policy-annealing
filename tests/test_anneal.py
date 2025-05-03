import torch

from lib.anneal import annealing_loss


def test_gradient_direction_and_symmetry():
    # Check gradient symmetry with reversed reward groups (2 groups of 2 trajectories)
    batch_size, group_size, num_actions = 4, 2, 2
    output = torch.zeros(batch_size, 1, num_actions, requires_grad=True)
    actions = torch.tensor([[0], [0], [0], [0]], dtype=torch.long)
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])  # Rewards: Group 1: [1, 0], Group 2: [0, 1] (reversed)
    done_mask = torch.tensor([[False], [False], [False], [False]])

    loss = annealing_loss(output, actions, rewards, done_mask, 1.0, group_size)
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
    done_mask = torch.tensor([[False, True], [False, False]])  # Done Mask: Mask out step 1 of trajectory 0

    loss = annealing_loss(softmax_output, actions, rewards, done_mask, 1.0, group_size, apply_softmax=False)
    softmax_output.retain_grad()
    loss.backward()

    assert softmax_output.grad is not None

    print(f"softmax_output.grad: {softmax_output.grad}")

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
    done_mask = torch.tensor([[False], [False]])

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

    loss = annealing_loss(output, actions, rewards, done_mask, temperature, group_size)

    # Loss should be close to zero
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
