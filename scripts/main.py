import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import math
import os
from argparse import ArgumentParser, Namespace
from datetime import datetime

import torch

from lib.anneal_grouped import anneal_grouped
from lib.anneal_value_function import anneal_value_function
from lib.grouped_environments import GroupedEnvironments
from lib.model import get_model, get_temperature
from lib.sample import sample_batch_episode
from lib.tracking import RenderValue, get_git_info, save_run


def main(args: Namespace) -> None:
    timestamp = datetime.now().isoformat().replace(":", "-").replace(".", "-")
    git_info = get_git_info()
    run_path = Path(__file__).parent.parent / "runs" / timestamp
    print(f"Run data will be stored in {run_path}")

    all_step_rewards: list[list[float]] = []
    for run in range(args.num_runs):
        print(f"Running experiment {run + 1} of {args.num_runs}:")
        step_rewards = run_experiment(args, run_path / f"{run:03d}")
        all_step_rewards.append(step_rewards)
        print()

    save_run(run_path / "experiment.json", args, git_info, all_step_rewards)


def run_experiment(args: Namespace, run_path: Path) -> list[float]:
    render_path = run_path / "training.mp4" if args.render == "full" else None
    envs = GroupedEnvironments(args.env_name, args.group_size, args.batch_size, not args.disable_group_initialization, render_path=render_path)

    # Initialize policy model and optionally a value model
    # Use common optimizer for both models
    policy = get_model(envs.num_observations, envs.num_actions, hidden=[32, 32])
    value = get_model(envs.num_observations, 1, hidden=[32, 32]) if args.value_function in ["difference", "direct"] else None
    if args.load_models is not None:
        policy.load_state_dict(torch.load(Path(args.load_models) / "policy.pth"))
        if value is not None:
            value.load_state_dict(torch.load(Path(args.load_models) / "value.pth"))
    params = list(policy.parameters()) + (list(value.parameters()) if value is not None else [])
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, betas=(0.5, 0.99))

    # Run annealing
    total_timesteps = 0
    step_rewards: list[float] = []
    render_first_obs: RenderValue | None = None
    for step in range(args.anneal_steps):
        temp = get_temperature(args.temp_start, args.temp_end, step / args.anneal_steps)

        # Sample batch
        observations, actions, rewards, terminated_mask, truncated_mask = sample_batch_episode(policy, envs)

        # Render first observation
        if args.render in ["plots", "full"] and value is not None:
            if render_first_obs is None:
                first_obs_path = run_path / "first_observation_value.html"
                render_first_obs = RenderValue(
                    "First episode over annealing steps",
                    first_obs_path,
                    observations[0],
                    actions[0],
                    rewards[0],
                    terminated_mask[0],
                    truncated_mask[0],
                )
            render_first_obs.update(policy, value, temp, args.discount_factor)

        # Anneal
        if args.value_function == "grouped":
            loss = anneal_grouped(policy, observations, actions, rewards, terminated_mask, truncated_mask, optimizer, temp, args.clip_eps, args.group_size, args.optim_steps)  # fmt: skip
        else:
            anneal_render_path = run_path / "value_over_steps" / f"{step:03d}.html" if args.render in ["plots", "full"] else None
            loss = anneal_value_function(policy, value, observations, actions, rewards, terminated_mask, truncated_mask, optimizer, temp, args.clip_eps, args.discount_factor, args.optim_steps, anneal_render_path)  # fmt: skip

        # Log stats
        total_timesteps += torch.sum(torch.logical_not(torch.logical_or(terminated_mask, truncated_mask))).item()
        total_samples = step * len(observations)
        valid_mask = torch.logical_not(torch.logical_or(terminated_mask, truncated_mask))
        mean_reward = torch.mean(torch.sum(rewards * valid_mask, dim=1))
        ep_length = torch.mean(torch.sum(valid_mask, dim=1, dtype=torch.float32))
        steps_formatted = f"[{step}/{args.anneal_steps} ({total_samples} samples) (timesteps {total_timesteps})]"
        stats_formatted = f"reward - {mean_reward:.2f}, eplength - {ep_length:.2f}, avg_diff - {math.sqrt(loss[0]):.2f}, temperature - {temp:.4}"  # fmt: skip
        print(f"Annealing {steps_formatted}: {stats_formatted}")
        step_rewards.append(mean_reward)

        # Save models
        step_path = run_path / "models" / f"{step:03d}"
        os.makedirs(step_path, exist_ok=True)
        torch.save(policy.state_dict(), step_path / "policy.pth")
        if value is not None:
            torch.save(value.state_dict(), step_path / "value.pth")

    # Close the first observation renderer if it was created
    if render_first_obs is not None and value is not None:
        render_first_obs.update(policy, value, args.temp_end, args.discount_factor)
        render_first_obs.close()

    return step_rewards


def parse_args() -> Namespace:
    # fmt: off
    parser = ArgumentParser()

    # Environment arguments
    parser.add_argument("--env-name", type=str, default="CartPole-v1", help="The name of the environment to run")

    # Optimization arguments
    parser.add_argument("--value-function", type=str, default="grouped", help="The type of value function to use", choices=["grouped", "difference", "direct"])
    parser.add_argument("--discount-factor", type=float, default=0.985, help="The gamma discount factor for the value function")
    parser.add_argument("--anneal-steps", type=int, default=100, help="The number of annealings to run (not related to environment steps)")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="The learning rate for the optimizer")
    parser.add_argument("--temp-start", type=float, default=2, help="The initial temperature of the annealing algorithm")
    parser.add_argument("--temp-end", type=float, default=2, help="The final temperature of the annealing algorithm")
    parser.add_argument("--clip-eps", type=float, default=100000.0, help="The clipping epsilon for the policy (by default, no clipping)")
    parser.add_argument("--group-size", type=int, default=8, help="The number of environments in each group with identical environment seeds")
    parser.add_argument("--disable-group-initialization", action="store_true", help="Disables common seeds for each group. Sets group size to batch size")
    parser.add_argument("--batch-size", type=int, default=32, help="Total number of environments to run in parallel (batch_size = group_size * (number of groups))")
    parser.add_argument("--optim-steps", type=int, default=30, help="The number of optimization steps within each annealing step")

    # Validation arguments
    parser.add_argument("--render", type=str, choices=["plots", "full"], help="Create visualizations of the training process: 'plots' for value plots only, 'full' for all visualizations")

    # Experiment arguments
    parser.add_argument("--load-models", type=str, help="Load models from the given path")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of experiment runs to perform")
    args = parser.parse_args()
    # fmt: on

    # Checks
    if args.value_function in ["difference", "direct"]:
        # Environments with same group have same seed, but we want different seeds for each environment
        print("Setting group size to 1 as value is modeled with a neural network")
        args.group_size = 1
        args.disable_group_initialization = True

    return args


if __name__ == "__main__":
    main(parse_args())
