import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

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
