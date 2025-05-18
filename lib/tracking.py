import json
import os
import subprocess
import math
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Optional
from argparse import Namespace

import numpy as np
import plotly.graph_objects as go
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
        self.sample_width = 600   # Default width
        
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
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            os.makedirs(render_path.parent, exist_ok=True)
            self.video_writer = cv2.VideoWriter(
                str(render_path),
                fourcc,
                30,  # FPS
                (self.grid_width, self.grid_height)
            )
    
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
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(
                        str(self.render_path),
                        fourcc,
                        30,  # FPS
                        (self.grid_width, self.grid_height)
                    )
            
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
                        render = self.last_valid_frames[i].copy()
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
                    -1  # Fill
                )
                
                # Draw the text
                cv2.putText(
                    render, 
                    reward_text, 
                    (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255),  # White
                    1, 
                    cv2.LINE_AA
                )
                
                # Place render in the grid
                grid_frame[y_start:y_end, x_start:x_end] = render
        
        # Write the frame
        self.video_writer.write(cv2.cvtColor(grid_frame, cv2.COLOR_RGB2BGR))
        self.last_frame = cv2.cvtColor(grid_frame, cv2.COLOR_RGB2BGR)
    
    def close(self):
        """Release the video writer resources."""
        if self.video_writer is not None:
            self.video_writer.release()