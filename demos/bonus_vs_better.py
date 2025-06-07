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
TEMPERATURE = 0.3
LEARNING_RATE = 0.2
BATCH_SIZE = 4
NUM_TRAINING_STEPS = 300


# --- Policy Model ---
# Since there is no state, the policy is just a learnable set of logits.
class Policy(nn.Module):
    """A simple policy model with learnable logits for each action."""

    def __init__(self, num_actions):
        super().__init__()
        # Initialize logits to zeros for an initial uniform distribution
        self.logits = nn.Parameter(torch.zeros(num_actions))

    def forward(self):
        """Returns the probability distribution over actions."""
        return torch.softmax(self.logits, dim=-1)


# --- Main Simulation Logic ---


def run_simulation():
    """
    Runs the simulation for both algorithms and returns the history of their policies.
    """
    # Initialize a separate policy and optimizer for each algorithm
    gpa_policy = Policy(NUM_ACTIONS)
    pg_policy = Policy(NUM_ACTIONS)

    gpa_optimizer = optim.SGD(gpa_policy.parameters(), lr=LEARNING_RATE)
    pg_optimizer = optim.SGD(pg_policy.parameters(), lr=LEARNING_RATE)

    # Store probability history for visualization
    gpa_probs_history = []
    pg_probs_history = []

    print("Starting simulation...")
    for step in range(NUM_TRAINING_STEPS):
        # Record the policy probabilities for visualization
        gpa_probs_history.append(gpa_policy().detach().numpy())
        pg_probs_history.append(pg_policy().detach().numpy())

        # --- 1. Grouped Policy Annealing (GPA) Step ---
        gpa_optimizer.zero_grad()
        gpa_probs = gpa_policy()
        gpa_actions_sampled = torch.multinomial(gpa_probs, num_samples=BATCH_SIZE, replacement=True)

        # Get the rewards and probabilities for the actions we just sampled
        gpa_rewards_sampled = REWARDS[gpa_actions_sampled]
        gpa_probs_sampled = gpa_probs[gpa_actions_sampled]

        # Calculate V' for each trajectory in the batch
        # V' = R - T * log(p)
        gpa_values = gpa_rewards_sampled - TEMPERATURE * torch.log(gpa_probs_sampled)

        # The loss aims to make all V' in the batch equal.
        # We use the mean squared error from the batch's average V'.
        gpa_mean_value = gpa_values.mean().detach()  # Detach to treat it as a constant target
        gpa_loss = ((gpa_values - gpa_mean_value) ** 2).mean()

        gpa_loss.backward()
        gpa_optimizer.step()

        # --- 2. Policy Gradients + Entropy Bonus (PG+EB) Step ---
        pg_optimizer.zero_grad()
        pg_probs = pg_policy()
        pg_actions_sampled = torch.multinomial(pg_probs, num_samples=BATCH_SIZE, replacement=True)

        # Get rewards and log-probabilities for the sampled actions
        pg_rewards_sampled = REWARDS[pg_actions_sampled]
        pg_log_probs_sampled = torch.log(pg_probs[pg_actions_sampled])

        # Use the batch average reward as the baseline for centering
        baseline = pg_rewards_sampled.mean()
        advantages = pg_rewards_sampled - baseline

        # Policy Gradient Loss: -E[log(pi(a|s)) * (R - baseline)]
        # We want to MAXIMIZE this, so we MINIMIZE its negative.
        policy_gradient_loss = -(pg_log_probs_sampled * advantages.detach()).mean()

        # Entropy Bonus Loss: -T * H(pi)
        # We want to MAXIMIZE entropy, so we MINIMIZE its negative.
        # Entropy H(pi) = -sum(p * log(p))
        entropy = -torch.sum(pg_probs * torch.log(pg_probs))
        entropy_loss = -TEMPERATURE * entropy

        # The total loss is a combination of exploiting rewards and encouraging exploration
        pg_loss = policy_gradient_loss + entropy_loss

        pg_loss.backward()
        pg_optimizer.step()

        if (step + 1) % 25 == 0:
            print(f"Step {step+1}/{NUM_TRAINING_STEPS} complete.")

    print("Simulation finished.")
    return np.array(gpa_probs_history), np.array(pg_probs_history)


def create_animation(gpa_history, pg_history):
    """
    Creates and displays a Plotly animated bar chart of the policy probabilities.
    """
    print("Generating animation...")
    # Create figure with initial data (step 0)
    fig = go.Figure(
        data=[
            go.Bar(name="Grouped Policy Annealing", x=ACTION_LABELS, y=gpa_history[0]),
            go.Bar(name="PG + Entropy Bonus", x=ACTION_LABELS, y=pg_history[0]),
        ]
    )

    # Create frames for the animation
    frames = []
    for i in range(1, NUM_TRAINING_STEPS):
        frame = go.Frame(name=str(i), data=[go.Bar(y=gpa_history[i]), go.Bar(y=pg_history[i])])
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
        title="Policy Annealing vs. Policy Gradients + Entropy Bonus",
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
    gpa_history, pg_history = run_simulation()
    create_animation(gpa_history, pg_history)
