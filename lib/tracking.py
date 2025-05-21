import json
import math
import os
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from pydantic import BaseModel


class ExperimentRun(BaseModel):
    """Model for storing experiment run data."""

    git_info: dict[str, Any]
    parameters: dict[str, Any]
    rewards: list[list[float]]


def save_run(run_path: Path, args: Namespace, git_info: dict[str, Any], rewards: list[list[float]]) -> None:
    """
    Save experiment run data to a JSON file.

    Args:
        run_path: Path to the experimentrun file
        args: Experiment parameters
        rewards: List of reward trajectories from multiple runs
    """
    params = vars(args)
    run = ExperimentRun(git_info=git_info, parameters=params, rewards=rewards)
    os.makedirs(run_path.parent, exist_ok=True)
    run_path.write_text(run.model_dump_json(indent=2))


def get_git_info() -> dict[str, Any]:
    """Get git information."""
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip(),
        "commit-dirty": subprocess.check_output(["git", "status", "--porcelain"]).decode("utf-8").strip(),
        "commit-message": subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode("utf-8").strip(),
        "commit-date": subprocess.check_output(["git", "log", "-1", "--pretty=%ad"]).decode("utf-8").strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"]).decode("utf-8").strip(),
    }


def load_run(run_path: Path) -> ExperimentRun:
    """
    Load experiment run data from a run file.

    Args:
        run_path: Path to the experiment run file

    Returns:
        ExperimentRun object containing the loaded data
    """
    data = json.loads(run_path.read_text())
    return ExperimentRun.model_validate(data)


def plot_percentiles(runs: list[ExperimentRun], percentiles: list[float] = [20.0, 50.0, 80.0]) -> None:
    """
    Plot reward percentiles from multiple experiment runs.

    Args:
        run_paths: List of paths to experiment JSON files
        percentiles: List of percentiles to plot (default: [20, 50, 80])
    """
    fig = make_subplots(rows=1, cols=1)

    colors = [
        "rgb(31, 119, 180)",
        "rgb(255, 127, 14)",
        "rgb(44, 160, 44)",
        "rgb(214, 39, 40)",
        "rgb(148, 103, 189)",
        "rgb(140, 86, 75)",
        "rgb(227, 119, 194)",
        "rgb(127, 127, 127)",
        "rgb(188, 189, 34)",
        "rgb(23, 190, 207)",
    ]

    for i, run in enumerate(runs):
        color = colors[i % len(colors)]
        rewards_array = np.array(run.rewards)
        exp_name = f"Experiment {i + 1}"
        steps = list(range(rewards_array.shape[1]))

        for percentile in percentiles:
            percentile_values = np.percentile(rewards_array, percentile, axis=0)
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=percentile_values,
                    mode="lines",
                    name=f"{exp_name} - p{percentile}",
                    line=dict(color=color),
                    legendgroup=exp_name,
                )
            )

    fig.update_layout(
        title="Reward Percentiles Across Experiments",
        xaxis_title="Annealing Step",
        yaxis_title="Reward",
        legend_title="Experiments",
        hovermode="x unified",
    )

    fig.show()


class RenderEpisodes:
    """
    Renders training episodes from a batch of environments into a single mp4 file.
    Creates a grid layout of environments, all rendered simultaneously.
    """

    def __init__(self, render_path: Optional[Path], batch_size: int, dummy_env):
        """
        Initialize the episode renderer.

        Args:
            render_path: Path to save the rendered video (None to disable rendering)
            batch_size: Number of environments in the batch
            dummy_env: A dummy environment instance to get render resolution from
        """
        self.render_path = render_path
        self.batch_size = batch_size

        # Determine optimal grid dimensions (as close to square as possible)
        self.grid_cols = math.ceil(math.sqrt(batch_size))
        self.grid_rows = math.ceil(batch_size / self.grid_cols)

        # For environments like CartPole, use standard dimensions if we can't render yet
        # These will be adjusted on the first step when we have actual renders
        self.sample_height = 400  # Default height
        self.sample_width = 600  # Default width

        # Try to get sample dimensions if possible
        try:
            # Get render resolution from a sample - reset first to avoid OrderEnforcer error
            dummy_env.reset()
            sample_render = dummy_env.render()
            self.sample_height, self.sample_width = sample_render.shape[:2]
        except Exception as e:
            print(f"Could not get render dimensions from dummy environment: {e}")
            print(f"Using default dimensions: {self.sample_width}x{self.sample_height}")

        # Calculate full grid resolution
        self.grid_width = self.sample_width * self.grid_cols
        self.grid_height = self.sample_height * self.grid_rows

        # Video writer setup
        self.video_writer = None
        self.accumulated_rewards = np.zeros(batch_size)
        self.last_frame = None
        self.dimensions_set = False

        # Keep track of the last valid frames for each environment
        self.last_valid_frames = [None] * batch_size

        if render_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            os.makedirs(render_path.parent, exist_ok=True)
            self.video_writer = cv2.VideoWriter(str(render_path), fourcc, 30, (self.grid_width, self.grid_height))  # FPS

    def reset(self):
        """
        Reset the renderer for a new batch of episodes.
        Add 30 freeze frames before continuing if last_frame exists.
        """
        if self.video_writer is not None and self.last_frame is not None:
            # Add 30 freeze frames showing the previous final state
            for _ in range(30):
                self.video_writer.write(self.last_frame)

        # Reset accumulated rewards and last valid frames
        self.accumulated_rewards = np.zeros(self.batch_size)
        self.last_valid_frames = [None] * self.batch_size

    def step(self, envs, rewards, valid_mask):
        """
        Render the current state of all environments in the batch.

        Args:
            envs: The vectorized environments
            rewards: Rewards from the current step
            valid_mask: Mask indicating which environments are still active
        """
        if self.video_writer is None:
            return

        # Update accumulated rewards for each environment
        self.accumulated_rewards += rewards

        # Get renders from all environments
        renders = envs.render()

        # If first render, check if we need to adjust dimensions and recreate video writer
        if not self.dimensions_set and renders is not None and len(renders) > 0:
            actual_height, actual_width = renders[0].shape[:2]

            # If actual dimensions differ from what we initialized with
            if actual_height != self.sample_height or actual_width != self.sample_width:
                print(f"Adjusting render dimensions from {self.sample_width}x{self.sample_height} to {actual_width}x{actual_height}")
                self.sample_height = actual_height
                self.sample_width = actual_width

                # Recalculate grid dimensions
                self.grid_width = self.sample_width * self.grid_cols
                self.grid_height = self.sample_height * self.grid_rows

                # Recreate video writer with correct dimensions
                if self.video_writer is not None:
                    self.video_writer.release()
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self.video_writer = cv2.VideoWriter(str(self.render_path), fourcc, 30, (self.grid_width, self.grid_height))  # FPS

            self.dimensions_set = True

        # Create the grid frame
        grid_frame = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)

        # Place each render in its grid position
        for i in range(self.batch_size):
            row = i // self.grid_cols
            col = i % self.grid_cols

            y_start = row * self.sample_height
            y_end = y_start + self.sample_height
            x_start = col * self.sample_width
            x_end = x_start + self.sample_width

            # Only process this slot if it's within batch_size
            if i < len(renders):
                # For valid environments, use the current render
                # For invalid environments, use the last valid render
                if valid_mask[i]:
                    render = renders[i].copy()
                    self.last_valid_frames[i] = render.copy()
                else:
                    # If this environment has just terminated, save its last valid frame
                    if self.last_valid_frames[i] is None and i < len(renders):
                        self.last_valid_frames[i] = renders[i].copy()

                    # Use the last valid frame if available
                    if self.last_valid_frames[i] is not None:
                        render = self.last_valid_frames[i].copy()  # type: ignore
                    else:
                        # If no last valid frame exists (shouldn't happen), use current frame
                        render = renders[i].copy()

                    # Apply gray effect
                    gray_render = cv2.cvtColor(render, cv2.COLOR_RGB2GRAY)
                    render = cv2.cvtColor(gray_render, cv2.COLOR_GRAY2RGB)

                    # Add red overlay with 20% opacity
                    red_overlay = np.zeros_like(render)
                    red_overlay[:, :] = [255, 0, 0]  # Red color
                    alpha = 0.2  # 20% opacity
                    render = cv2.addWeighted(render, 1 - alpha, red_overlay, alpha, 0)

                # Add accumulated reward text (in the bottom-right corner)
                reward_text = f"R: {self.accumulated_rewards[i]:.1f}"
                text_size = cv2.getTextSize(reward_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = self.sample_width - text_size[0] - 10
                text_y = self.sample_height - 10

                # Add a dark background for the text to be more visible
                text_bg_pad = 5
                cv2.rectangle(
                    render,
                    (text_x - text_bg_pad, text_y - text_size[1] - text_bg_pad),
                    (text_x + text_size[0] + text_bg_pad, text_y + text_bg_pad),
                    (0, 0, 0),
                    -1,  # Fill
                )

                # Draw the text
                cv2.putText(render, reward_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # White

                # Place render in the grid
                grid_frame[y_start:y_end, x_start:x_end] = render

        # Write the frame
        self.video_writer.write(cv2.cvtColor(grid_frame, cv2.COLOR_RGB2BGR))
        self.last_frame = cv2.cvtColor(grid_frame, cv2.COLOR_RGB2BGR)

    def close(self):
        """Release the video writer resources."""
        if self.video_writer is not None:
            self.video_writer.release()


class RenderValue:
    """
    Renders training value and other training data from a batch of environments into a single plotly animation (in a HTML container).
    The animation spans either the annealing steps or a single annealing episode.
    Only visualizes the valid part of the episode (where valid_mask is True).
    """

    def __init__(
        self, title: str, render_path: Path, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, valid_mask: torch.Tensor
    ):
        """
        Initialize the value renderer.

        Args:
            title: Title of the animation
            render_path: Path to save the rendered animation
            observations: Observations from the batch
            actions: Actions from the batch
            rewards: Rewards from the batch
            valid_mask: Mask indicating which steps are valid
        """
        self.title = title
        self.render_path = render_path
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.valid_mask = valid_mask

        # Find the episode length (where valid_mask changes from True to False)
        self.episode_length = self._get_episode_length(valid_mask)

        # Storage for annealing equation components
        self.lhs_values: list[torch.Tensor] = []
        self.rhs_values: list[torch.Tensor] = []
        self.discrepancy_values: list[torch.Tensor] = []

        # Create subplots with 4 rows
        self.fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Action Probabilities", "Rewards", "Value Function", "Annealing Equation Components"],
            vertical_spacing=0.1,
        )

        # Storage for animation frames
        self.frames: list[dict[str, Any]] = []
        self.step_count = 0

        # Track min/max values for y-axis scaling
        self.min_value = float("inf")
        self.max_value = float("-inf")
        self.min_reward = float("inf")
        self.max_reward = float("-inf")
        self.min_equation = float("inf")
        self.max_equation = float("-inf")

        # Set up figure with initial state
        self.fig.update_layout(title=title, width=800, height=1200, showlegend=True)  # Increased height for four subplots

    def _get_episode_length(self, mask: torch.Tensor) -> int:
        """
        Find the episode length by identifying where valid_mask changes from True to False.
        If all values are True, returns the full length.

        Args:
            mask: The valid_mask tensor

        Returns:
            The episode length
        """
        # If all values are True, return the full length
        if all(mask):
            return len(mask)

        # Find the first False value
        for i in range(len(mask)):
            if not mask[i]:
                return i

        # If we didn't find any False values, return the full length
        return len(mask)

    def update(self, policy: torch.nn.Module, value: torch.nn.Module, temp: float, discount_factor: float) -> None:
        """
        Store new frame of the animation with new policy and value functions.
        Only stores data for the valid part of the episode (where valid_mask is True).

        Args:
            policy: Policy function
            value: Value function
            temp: Temperature
            discount_factor: Discount factor
        """
        # Generate values
        with torch.no_grad():
            # Get value predictions and keep as tensor for equation calculations
            value_preds_tensor = value(self.observations).squeeze().detach()
            value_preds = value_preds_tensor.cpu().numpy()

            # Get policy outputs for calculation
            policy_outputs = policy(self.observations)

            # Calculate the annealing equation components
            # Compute log-probabilities (similar to value_annealing_loss function)
            log_probs = torch.log_softmax(policy_outputs, dim=1)
            selected_log_probs = torch.gather(log_probs, dim=1, index=self.actions.long().unsqueeze(1)).squeeze(1)

            # Apply rule: value is 0 after the episode ended
            values_masked = value_preds_tensor * self.valid_mask

            # Prepare V(s_{t+1})
            next_values = torch.cat([values_masked[1:], torch.zeros(1, device=values_masked.device)])

            # Left-hand side (LHS) of equation
            lhs = selected_log_probs * temp

            # Right-hand side (RHS) of equation
            value_target_difference = (discount_factor * next_values) - values_masked
            rhs = value_target_difference + self.rewards

            # Calculate the discrepancy
            discrepancy = lhs - rhs

            # Store the values for this step
            self.lhs_values.append(lhs.detach().cpu().numpy())
            self.rhs_values.append(rhs.detach().cpu().numpy())
            self.discrepancy_values.append(discrepancy.detach().cpu().numpy())

            # Also calculate probabilities for the standard graph
            probs = torch.softmax(policy_outputs, dim=1).detach().cpu().numpy()

            # Get probability of the actions that were actually taken
            action_indices = self.actions.long().cpu().numpy()
            taken_probs = np.array([probs[i, action_indices[i]] for i in range(len(self.observations))])

        # Get rewards and valid mask as numpy arrays
        rewards = self.rewards.cpu().numpy()

        # Only use the valid part of the episode (up to episode_length)
        valid_indices = list(range(self.episode_length))

        # Get valid parts of data
        valid_values = value_preds[: self.episode_length]
        valid_rewards = rewards[: self.episode_length]
        valid_probs = taken_probs[: self.episode_length]

        # Get valid parts of annealing equation components
        valid_lhs = self.lhs_values[-1][: self.episode_length]
        valid_rhs = self.rhs_values[-1][: self.episode_length]
        valid_discrepancy = self.discrepancy_values[-1][: self.episode_length]

        # Update min/max values for y-axis scaling
        self.min_value = min(self.min_value, valid_values.min())
        self.max_value = max(self.max_value, valid_values.max())
        self.min_reward = min(self.min_reward, valid_rewards.min())
        self.max_reward = max(self.max_reward, valid_rewards.max())

        # Update min/max values for equation components
        all_equation_values = np.concatenate([valid_lhs, valid_rhs, valid_discrepancy])
        if len(all_equation_values) > 0:
            self.min_equation = min(self.min_equation, np.min(all_equation_values))
            self.max_equation = max(self.max_equation, np.max(all_equation_values))

        # Create a dictionary to store the frame data
        frame_data = {
            "step": self.step_count,
            "temp": temp,
            "discount": discount_factor,
            "x": valid_indices,
            "values": valid_values,
            "rewards": valid_rewards,
            "action_probs": valid_probs,
            "lhs": valid_lhs,
            "rhs": valid_rhs,
            "discrepancy": valid_discrepancy,
        }

        # Store the frame
        self.frames.append(frame_data)
        self.step_count += 1

    def close(self) -> None:
        """
        Save the animated plot.
        Only shows the valid part of the episode (where valid_mask is True).
        """
        if not self.frames:
            print("No frames to render.")
            return

        # Create frames for animation
        animation_frames = []
        for frame in self.frames:
            # Get data for this frame (already filtered to only valid steps)
            x = frame["x"]
            values = frame["values"]
            rewards = frame["rewards"]
            action_probs = frame["action_probs"]

            # Create traces for this frame
            frame_traces = []

            # Value trace (third subplot)
            value_trace = go.Scatter(
                x=x,
                y=values,
                mode="lines+markers",
                name="Predicted Values",
                line=dict(color="blue"),
                marker=dict(size=8),
                showlegend=True,
                legendgroup="values",
            )

            # Reward trace (second subplot)
            reward_trace = go.Scatter(
                x=x,
                y=rewards,
                mode="lines+markers",
                name="Rewards",
                line=dict(color="orange"),
                marker=dict(size=8),
                showlegend=True,
                legendgroup="rewards",
            )

            # Action probability trace (first subplot)
            prob_trace = go.Scatter(
                x=x,
                y=action_probs,
                mode="lines+markers",
                name="Action Probabilities",
                line=dict(color="purple"),
                marker=dict(size=8),
                showlegend=True,
                legendgroup="probs",
            )

            # Get annealing equation components from the frame data
            lhs = frame["lhs"]
            rhs = frame["rhs"]
            discrepancy = frame["discrepancy"]

            # LHS (left-hand side) trace
            lhs_trace = go.Scatter(
                x=x,
                y=lhs,
                mode="lines+markers",
                name="LHS (log_probs * temp)",
                line=dict(color="green"),
                marker=dict(size=8),
                showlegend=True,
                legendgroup="equation",
            )

            # RHS (right-hand side) trace
            rhs_trace = go.Scatter(
                x=x,
                y=rhs,
                mode="lines+markers",
                name="RHS ((discount * next_value) - value + reward)",
                line=dict(color="red"),
                marker=dict(size=8),
                showlegend=True,
                legendgroup="equation",
            )

            # Discrepancy trace
            discrepancy_trace = go.Scatter(
                x=x,
                y=discrepancy,
                mode="lines+markers",
                name="Discrepancy (LHS - RHS)",
                line=dict(color="black", dash="dash"),
                marker=dict(size=8),
                showlegend=True,
                legendgroup="equation",
            )

            # Collect all traces with their corresponding subplot positions
            frame_traces = [
                (prob_trace, 1, 1),  # Probabilities in row 1
                (reward_trace, 2, 1),  # Rewards in row 2
                (value_trace, 3, 1),  # Values in row 3
                (lhs_trace, 4, 1),  # LHS in row 4
                (rhs_trace, 4, 1),  # RHS in row 4
                (discrepancy_trace, 4, 1),  # Discrepancy in row 4
            ]

            # Create animation frame
            animation_frame: dict[str, Any] = {"data": [], "name": f"Step {frame['step']}"}

            # Add traces to animation frame with proper subplot placement
            for trace, row, col in frame_traces:
                # Create a new trace with the same properties instead of using copy()
                trace_dict = trace.to_plotly_json()
                trace_dict.update({"xaxis": f"x{row}" if row > 1 else "x", "yaxis": f"y{row}" if row > 1 else "y"})
                animation_frame["data"].append(trace_dict)

            # Add frame title with current temperature and discount
            animation_frame["layout"] = {
                "title": f"{self.title} - Step {frame['step']}, Temp: {frame['temp']:.4f}, Discount: {frame['discount']:.4f}"
            }

            animation_frames.append(animation_frame)

        # Create animation
        self.fig.frames = animation_frames

        # Add initial traces (will be updated by frames)
        first_frame = self.frames[0]
        x = first_frame["x"]
        values = first_frame["values"]
        rewards = first_frame["rewards"]
        action_probs = first_frame["action_probs"]
        lhs = first_frame["lhs"]
        rhs = first_frame["rhs"]
        discrepancy = first_frame["discrepancy"]

        # Add action probability trace to the first (top) subplot
        self.fig.add_trace(
            go.Scatter(x=x, y=action_probs, mode="lines+markers", name="Action Probabilities", line=dict(color="purple"), marker=dict(size=8)),
            row=1,
            col=1,
        )

        # Add reward trace to the second (middle) subplot
        self.fig.add_trace(
            go.Scatter(x=x, y=rewards, mode="lines+markers", name="Rewards", line=dict(color="orange"), marker=dict(size=8)), row=2, col=1
        )

        # Add value trace to the third subplot
        self.fig.add_trace(
            go.Scatter(x=x, y=values, mode="lines+markers", name="Predicted Values", line=dict(color="blue"), marker=dict(size=8)), row=3, col=1
        )

        # Add LHS trace to the fourth subplot
        self.fig.add_trace(
            go.Scatter(x=x, y=lhs, mode="lines+markers", name="LHS (log_probs * temp)", line=dict(color="green"), marker=dict(size=8)), row=4, col=1
        )

        # Add RHS trace to the fourth subplot
        self.fig.add_trace(
            go.Scatter(
                x=x, y=rhs, mode="lines+markers", name="RHS ((discount * next_value) - value + reward)", line=dict(color="red"), marker=dict(size=8)
            ),
            row=4,
            col=1,
        )

        # Add discrepancy trace to the fourth subplot
        self.fig.add_trace(
            go.Scatter(
                x=x, y=discrepancy, mode="lines+markers", name="Discrepancy (LHS - RHS)", line=dict(color="black", dash="dash"), marker=dict(size=8)
            ),
            row=4,
            col=1,
        )

        # Add slider for navigation through steps
        steps = []
        for i, frame in enumerate(self.frames):
            step = {
                "args": [
                    [f"Step {frame['step']}"],
                    {"frame": {"duration": 300, "redraw": True}, "mode": "immediate", "transition": {"duration": 300}},
                ],
                "label": f"Step {frame['step']}",
                "method": "animate",
            }
            steps.append(step)

        sliders = [
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {"font": {"size": 16}, "prefix": "Annealing Step: ", "visible": True, "xanchor": "right"},
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": steps,
            }
        ]

        # Add play and pause buttons
        updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True, "transition": {"duration": 300}}],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ]

        # Update layout with animation controls
        self.fig.update_layout(updatemenus=updatemenus, sliders=sliders)

        # Update axes labels and ranges
        # First subplot - Action Probabilities
        self.fig.update_xaxes(title_text="Step", row=1, col=1)
        self.fig.update_yaxes(
            title_text="Probability",
            row=1,
            col=1,
            # Fixed range from 0 to 1 for probabilities
            range=[0, 1],
        )

        # Second subplot - Rewards
        self.fig.update_xaxes(title_text="Step", row=2, col=1)
        self.fig.update_yaxes(
            title_text="Reward",
            row=2,
            col=1,
            # Set y-axis range with some padding
            range=[self.min_reward - 0.1 * (self.max_reward - self.min_reward), self.max_reward + 0.1 * (self.max_reward - self.min_reward)],
        )

        # Third subplot - Value Function
        self.fig.update_xaxes(title_text="Step", row=3, col=1)
        self.fig.update_yaxes(
            title_text="Value",
            row=3,
            col=1,
            # Set y-axis range with some padding
            range=[self.min_value - 0.1 * (self.max_value - self.min_value), self.max_value + 0.1 * (self.max_value - self.min_value)],
        )

        # Fourth subplot - Annealing Equation Components
        self.fig.update_xaxes(title_text="Step", row=4, col=1)
        self.fig.update_yaxes(
            title_text="Value",
            row=4,
            col=1,
            # Set y-axis range with some padding
            range=[
                self.min_equation - 0.1 * (self.max_equation - self.min_equation),
                self.max_equation + 0.1 * (self.max_equation - self.min_equation),
            ],
        )

        # Make sure the directory exists
        os.makedirs(self.render_path.parent, exist_ok=True)

        # Save the figure to HTML
        self.fig.write_html(self.render_path, include_plotlyjs="cdn", full_html=True, auto_open=False)
