import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import math
from argparse import ArgumentParser, Namespace
from datetime import datetime

import torch

from lib.anneal_grouped import anneal_grouped
from lib.anneal_value_function import anneal_value_function
from lib.grouped_environments import GroupedEnvironments
from lib.model import get_model, get_temperature
from lib.sample import sample_batch_episode
from lib.tracking import get_git_info, save_run


def main(args: Namespace) -> None:
    timestamp = datetime.now().isoformat().replace(":", "-").replace(".", "-")
    git_info = get_git_info()
    run_path = Path(__file__).parent.parent / "runs" / timestamp / "experiment.json"

    all_step_rewards: list[list[float]] = []
    for _ in range(args.num_runs):
        print(f"Running experiment {_ + 1} of {args.num_runs}:")
        step_rewards = run_experiment(args)
        all_step_rewards.append(step_rewards)
        print()

    save_run(run_path, args, git_info, all_step_rewards)


def run_experiment(args: Namespace) -> list[float]:
    envs = GroupedEnvironments(args.env_name, args.group_size, args.batch_size, not args.disable_group_initialization, render=args.render)

    # Initialize policy model and optionally a value model
    # Use common optimizer for both models
    policy = get_model(envs.num_observations, envs.num_actions, hidden=[32])
    value = get_model(envs.num_observations, 1, hidden=[32]) if args.value_model == "network" else None
    params = list(policy.parameters()) + (list(value.parameters()) if value is not None else [])
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Run annealing
    step_rewards: list[float] = []
    for step in range(args.anneal_steps):
        # Sample batch
        observations, actions, rewards, valid_mask = sample_batch_episode(policy, envs)

        # Anneal
        temp = get_temperature(args.temp_start, args.temp_end, step / args.anneal_steps)
        if args.value_model == "grouped":
            loss = anneal_grouped(policy, observations, actions, rewards, valid_mask, optimizer, temp, args.clip_eps, args.group_size, args.optim_steps)  # fmt: skip
        else:
            loss = anneal_value_function(policy, value, observations, actions, rewards, valid_mask, optimizer, temp, args.clip_eps, args.discount_factor, args.optim_steps)  # fmt: skip

        # Log stats
        total_samples = step * len(observations)
        mean_reward = torch.mean(torch.sum(rewards * valid_mask, dim=1))
        ep_length = torch.mean(torch.sum(valid_mask, dim=1, dtype=torch.float32))
        steps_formatted = f"[{step}/{args.anneal_steps} ({total_samples} samples)]"
        stats_formatted = f"reward - {mean_reward:.2f}, eplength - {ep_length:.2f}, avg_diff - {math.sqrt(loss[0]):.2f}, temperature - {temp:.4}"  # fmt: skip
        print(f"Annealing {steps_formatted}: {stats_formatted}")
        step_rewards.append(mean_reward)

    return step_rewards


def parse_args() -> Namespace:
    # fmt: off
    parser = ArgumentParser()

    # Environment arguments
    parser.add_argument("--env_name", type=str, default="CartPole-v1", help="The name of the environment to run")

    # Optimization arguments
    parser.add_argument("--value-model", type=str, default="grouped", help="The type of value function to use", choices=["grouped", "network"])
    parser.add_argument("--discount-factor", type=float, default=0.98, help="The gamma discount factor for the value function")
    parser.add_argument("--anneal_steps", type=int, default=100, help="The number of annealings to run (not related to environment steps)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="The learning rate for the optimizer")
    parser.add_argument("--temp_start", type=float, default=0.5, help="The initial temperature of the annealing algorithm")
    parser.add_argument("--temp_end", type=float, default=0.5, help="The final temperature of the annealing algorithm")
    parser.add_argument("--clip_eps", type=float, default=100000.0, help="The clipping epsilon for the policy (by default, no clipping)")
    parser.add_argument("--group_size", type=int, default=8, help="The number of environments in each group with identical environment seeds")
    parser.add_argument("--disable_group_initialization", action="store_true", help="Disables common seeds for each group. Sets group size to batch size")
    parser.add_argument("--batch_size", type=int, default=32, help="Total number of environments to run in parallel (batch_size = group_size * (number of groups))")
    parser.add_argument("--optim_steps", type=int, default=30, help="The number of optimization steps within each annealing step")

    # Validation arguments
    parser.add_argument("--render", action="store_true", help="Create various visualizations of the training process into `runs/` directory")

    # Experiment arguments
    parser.add_argument("--num_runs", type=int, default=1, help="Number of experiment runs to perform")
    args = parser.parse_args()
    # fmt: on

    # Checks
    if args.value_model == "network":
        # Environments with same group have same seed, but we want different seeds for each environment
        print("Setting group size to 1 as value is modeled with a neural network")
        args.group_size = 1
        args.disable_group_initialization = True

    return args


if __name__ == "__main__":
    main(parse_args())
