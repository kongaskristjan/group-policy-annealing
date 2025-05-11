import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel


class ExperimentRun(BaseModel):
    """Model for storing experiment run data."""

    parameters: dict[str, Any]
    rewards: list[list[float]]


def save_run(run_path: Path, args: Namespace, rewards: list[list[float]]) -> None:
    """
    Save experiment run data to a JSON file.

    Args:
        run_path: Path to the experimentrun file
        args: Experiment parameters
        rewards: List of reward trajectories from multiple runs
    """
    params = vars(args)
    run = ExperimentRun(parameters=params, rewards=rewards)
    os.makedirs(run_path.parent, exist_ok=True)
    run_path.write_text(run.model_dump_json(indent=2))


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
