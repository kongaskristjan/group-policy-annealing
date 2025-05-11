import math
from argparse import ArgumentParser, Namespace

import torch

from lib.anneal import anneal_batch_episode, get_temperature
from lib.grouped_environments import GroupedEnvironments
from lib.model import get_model
from lib.sample import render_episode, sample_batch_episode, validate
from lib.tracking import save_run


def main(args: Namespace) -> None:
    all_step_rewards: list[list[float]] = []
    for _ in range(args.num_runs):
        step_rewards = run_experiment(args)
        all_step_rewards.append(step_rewards)
    save_run(args, all_step_rewards)


def run_experiment(args: Namespace) -> list[float]:
    envs = GroupedEnvironments(args.env_name, args.group_size, args.batch_size, not args.disable_group_initialization)
    model = get_model(envs.num_observations, envs.num_actions, hidden=[32])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    step_rewards: list[float] = []
    for step in range(args.anneal_steps):
        if args.render_freq > 0 and step % args.render_freq == 0:
            render_episode(model, args.env_name)

        if args.val_freq > 0 and step % args.val_freq == 0:
            val_reward = validate(model, args.env_name, args.val_batch)
            print(f"Validation [{step} ({step * args.batch_size} samples)]: mean_reward - {val_reward:.2f}")

        observations, actions, rewards, done_mask = sample_batch_episode(model, envs)
        temp = get_temperature(args.temp_start, args.temp_end, step / args.anneal_steps)
        loss = anneal_batch_episode(
            model, observations, actions, rewards, done_mask, optimizer, temp, args.clip_eps, args.group_size, args.optim_steps
        )

        total_samples = step * args.batch_size
        ep_length = torch.mean(torch.sum(torch.logical_not(done_mask), dim=1, dtype=torch.float32))
        steps_formatted = f"[{step}/{args.anneal_steps} ({total_samples} samples)]"
        stats_formatted = f"reward - {torch.mean(rewards):.2f}, eplength - {ep_length:.2f}, avg_diff - {math.sqrt(loss[0]):.2f}, temperature - {temp:.4}"  # fmt: skip
        print(f"Annealing {steps_formatted}: {stats_formatted}")
        step_rewards.append(torch.mean(rewards))

    return step_rewards

def parse_args() -> Namespace:
    # fmt: off
    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole-v1", help="The name of the environment to run")
    parser.add_argument("--anneal_steps", type=int, default=100, help="The number of annealings to run (not related to environment steps)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="The learning rate for the optimizer")
    parser.add_argument("--temp_start", type=float, default=0.5, help="The initial temperature of the annealing algorithm")
    parser.add_argument("--temp_end", type=float, default=0.5, help="The final temperature of the annealing algorithm")
    parser.add_argument("--clip_eps", type=float, default=100000.0, help="The clipping epsilon for the policy (by default, no clipping)")
    parser.add_argument("--group_size", type=int, default=8, help="The number of environments in each group with identical environment seeds")
    parser.add_argument("--disable_group_initialization", action="store_true", help="Disables common seeds for each group. Sets group size to batch size.")
    parser.add_argument("--batch_size", type=int, default=32, help="Total number of environments to run in parallel (batch_size = group_size * (number of groups))")
    parser.add_argument("--optim_steps", type=int, default=30, help="The number of optimization steps within each annealing step")

    # Validation arguments
    parser.add_argument("--val_freq", type=int, default=0, help="Frequency of validation in terms of annealing steps (0 to disable)")
    parser.add_argument("--val_batch", type=int, default=32, help="Number of environments to run during validation")
    parser.add_argument("--render_freq", type=int, default=0, help="Frequency of rendering in terms of annealing steps (0 to disable)")
    args = parser.parse_args()
    # fmt: on

    return args


if __name__ == "__main__":
    main(parse_args())
