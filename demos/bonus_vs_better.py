from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim

# --- Simulation Parameters ---

# Environment: 3-armed bandit with deterministic rewards
ACTION_LABELS = ["A (Reward=0)", "B (Reward=0)", "C (Reward=1)"]
REWARDS = torch.tensor([0.0, 0.0, 1.0])
NUM_ACTIONS = len(REWARDS)

# Hyperparameters
TEMPERATURE = 0.5
LEARNING_RATE = 1.0
BATCH_SIZE = 4
NUM_TRAINING_STEPS = 300


# --- Policy Model ---
# Since there is no state, the policy is just a learnable set of logits.
class Policy(nn.Module):
    """A simple policy model with learnable logits for each action."""

    def __init__(self, num_actions: int):
        super().__init__()
        # Initialize logits to zeros for an initial uniform distribution
        self.logits = nn.Parameter(torch.zeros(num_actions))

    def forward(self) -> torch.Tensor:
        """Returns the probability distribution over actions."""
        return torch.softmax(self.logits, dim=-1)


# --- Algorithm-Specific Loss Functions ---


def calculate_annealing_based_loss(
    probs: torch.Tensor, actions_sampled: torch.Tensor, rewards_sampled: torch.Tensor, use_policy_gradient: bool
) -> torch.Tensor:
    """Calculates the loss for either Policy Annealing (GPA) or Policy Gradients + Policy Annealing Regularization (PG+PA)."""
    probs_sampled = probs[actions_sampled]
    log_probs_sampled = torch.log(probs_sampled)

    # Calculate V' = R - T * log(p) for each trajectory
    values = rewards_sampled - TEMPERATURE * log_probs_sampled

    # Use V' as the advantage, centered around its batch mean
    mean_value = values.mean()
    if use_policy_gradient:
        # Policy Gradient Loss with V' as the advantage (includes annealing regularization)
        advantages = values - mean_value
        return -(log_probs_sampled * advantages.detach()).mean()

    # Regular Policy Annealing
    return ((values - mean_value) ** 2).mean()


def calculate_pg_eb_loss(probs: torch.Tensor, actions_sampled: torch.Tensor, rewards_sampled: torch.Tensor, use_entropy: bool) -> torch.Tensor:
    """Calculates the loss for Policy Gradients + Entropy Bonus (PG+EB)."""
    log_probs_sampled = torch.log(probs[actions_sampled])

    # Use the batch average reward as the baseline for centering
    baseline = rewards_sampled.mean()
    advantages = rewards_sampled - baseline

    # Policy Gradient Loss: -E[log(pi(a|s)) * (R - baseline)]
    policy_gradient_loss = -(log_probs_sampled * advantages.detach()).mean()

    # Entropy Bonus Loss: -T * H(pi) where H(pi) = -sum(p * log(p))
    if use_entropy:
        entropy = -torch.sum(probs * torch.log(probs))
        entropy_loss = -TEMPERATURE * entropy
        return policy_gradient_loss + entropy_loss
    return policy_gradient_loss


# --- Main Simulation Logic ---


def run_simulation() -> dict[str, np.ndarray]:
    """
    Runs the simulation for all algorithms and returns their policy histories.
    """
    # Define the algorithms to run, each with its own policy, optimizer, and loss function
    # fmt: off
    algorithms: list[dict[str, Any]] = [
        {"name": "Policy Gradients without regularization", "loss_fn": calculate_pg_eb_loss, "history": [], "kwargs": {"use_entropy": False}},
        {"name": "Policy Gradients + Entropy Bonus (Industry standard)", "loss_fn": calculate_pg_eb_loss, "history": [], "kwargs": {"use_entropy": True}},
        {"name": "Policy Gradients + Stable Entropy (Proposed algorithm)", "loss_fn": calculate_annealing_based_loss, "history": [], "kwargs": {"use_policy_gradient": True}},
        # {"name": "Grouped Policy Annealing", "loss_fn": calculate_annealing_based_loss, "history": [], "kwargs": {"use_policy_gradient": False}},
    ]
    # fmt: on

    # Initialize a separate policy and optimizer for each algorithm
    for alg in algorithms:
        alg["policy"] = Policy(NUM_ACTIONS)
        alg["optimizer"] = optim.SGD(alg["policy"].parameters(), lr=LEARNING_RATE)

    print("Starting simulation...")
    for step in range(NUM_TRAINING_STEPS):
        # First, record the current policy probabilities for all algorithms
        for alg in algorithms:
            probs = alg["policy"]().detach().numpy()
            alg["history"].append(probs)

        # Then, perform a training step for each algorithm
        for alg in algorithms:
            policy = alg["policy"]
            optimizer = alg["optimizer"]
            loss_fn = alg["loss_fn"]
            kwargs = alg["kwargs"]

            optimizer.zero_grad()

            # --- Common step: Sample actions and get rewards ---
            current_probs = policy()
            actions_sampled = torch.multinomial(current_probs, num_samples=BATCH_SIZE, replacement=True)
            rewards_sampled = REWARDS[actions_sampled]

            # --- Algorithm-specific step: Calculate loss ---
            loss = loss_fn(current_probs, actions_sampled, rewards_sampled, **kwargs)

            # --- Common step: Backpropagate and update weights ---
            loss.backward()
            optimizer.step()

        if (step + 1) % 25 == 0:
            print(f"Step {step+1}/{NUM_TRAINING_STEPS} complete.")

    print("Simulation finished.")

    # Return the histories as separate numpy arrays
    histories = {alg["name"]: np.array(alg["history"]) for alg in algorithms}

    return histories


def create_animation(histories: dict[str, np.ndarray]):
    """
    Creates and displays a Plotly animated bar chart of the policy probabilities.
    """
    print("Generating animation...")
    # Create figure with initial data (step 0)
    fig = go.Figure(data=[go.Bar(name=name, x=ACTION_LABELS, y=histories[name][0]) for name in histories])

    # Create frames for the animation
    frames = []
    for i in range(1, NUM_TRAINING_STEPS):
        frame = go.Frame(name=str(i), data=[go.Bar(y=histories[name][i]) for name in histories])
        frames.append(frame)

    fig.frames = frames

    # Configure animation settings
    animation_settings = {"frame": {"duration": 50, "redraw": True}, "transition": {"duration": 25, "easing": "linear"}}

    # Add play/pause button
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, animation_settings],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "showactive": False,
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ]
    )

    # Add a slider to navigate through steps
    sliders = [
        {
            "active": 0,
            "steps": [
                {
                    "label": f"Step {k}",
                    "method": "animate",
                    "args": [[f"{k}"], {"mode": "immediate", "frame": {"duration": 100, "redraw": True}, "transition": {"duration": 50}}],
                }
                for k in range(NUM_TRAINING_STEPS)
            ],
            "pad": {"t": 50, "b": 10},
            "len": 0.9,
            "x": 0.05,
            "y": 0,
            "yanchor": "top",
        }
    ]

    fig.update_layout(
        title="Entropy Bonus vs Stable Entropy (Proposed algorithm)",
        xaxis_title="Action",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        barmode="group",
        sliders=sliders,
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    fig.show()


if __name__ == "__main__":
    histories = run_simulation()
    create_animation(histories)
