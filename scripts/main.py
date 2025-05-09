import copy
import math
from argparse import ArgumentParser

import torch

from lib.anneal import anneal_batch_episode, get_temperature
from lib.grouped_environments import GroupedEnvironments
from lib.model import get_model
from lib.reward_bump_detector import RewardBumpDetector
from lib.sample import render_episode, sample_batch_episode, validate


def main(
    env_name: str,
    anneal_steps: int,
    learning_rate: float,
    temp_start: float,
    temp_end: float,
    clip_eps: float,
    group_size: int,
    enable_group_initialization: bool,
    batch_size: int,
    optim_steps: int,
    bump_threshold: float,
    initial_averaging_episodes: int,
    min_good_samples: int,
    val_freq: int,
    val_batch: int,
    render_freq: int,
) -> None:

    envs = GroupedEnvironments(env_name, group_size, batch_size, enable_group_initialization)
    reward_bump_detector = RewardBumpDetector(bump_threshold, initial_averaging_episodes, min_good_samples)
    model = get_model(envs.num_observations, envs.num_actions, hidden=[32])
    baseline_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for step in range(anneal_steps):
        if render_freq > 0 and step % render_freq == 0:
            render_episode(model, env_name)

        if val_freq > 0 and step % val_freq == 0:
            val_reward = validate(model, env_name, val_batch)
            print(f"Validation [{step} ({step * batch_size} samples)]: mean_reward - {val_reward:.2f}")

        observations, actions, rewards, done_mask = sample_batch_episode(model, envs)
        temp = get_temperature(temp_start, temp_end, step / anneal_steps)
        bump_detected = reward_bump_detector.update(torch.mean(rewards).item())
        if bump_detected:
            baseline_model = copy.deepcopy(model)
            print("Baseline model updated")
        loss = anneal_batch_episode(
            model, baseline_model, observations, actions, rewards, done_mask, optimizer, temp, clip_eps, group_size, optim_steps
        )

        total_samples = step * batch_size
        ep_length = torch.mean(torch.sum(torch.logical_not(done_mask), dim=1, dtype=torch.float32))
        steps_formatted = f"[{step}/{anneal_steps} ({total_samples} samples)]"
        stats_formatted = f"reward - {torch.mean(rewards):.2f}, eplength - {ep_length:.2f}, avg_diff - {math.sqrt(loss[0]):.2f}, temperature - {temp:.4}"  # fmt: skip
        print(f"Annealing {steps_formatted}: {stats_formatted}")


def parse_args() -> tuple[str, int, float, float, float, float, int, bool, int, int, float, int, int, int, int, int]:
    # fmt: off
    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole-v1", help="The name of the environment to run")
    parser.add_argument("--anneal_steps", type=int, default=30, help="The number of annealings to run (not related to environment steps)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="The learning rate for the optimizer")
    parser.add_argument("--temp_start", type=float, default=0.5, help="The initial temperature of the annealing algorithm")
    parser.add_argument("--temp_end", type=float, default=0.5, help="The final temperature of the annealing algorithm")
    parser.add_argument("--clip_eps", type=float, default=1.0, help="The clipping epsilon for the policy")
    parser.add_argument("--group_size", type=int, default=8, help="The number of environments in each group with identical environment seeds")
    parser.add_argument("--disable_group_initialization", action="store_true", help="Disables common seeds for each group. Sets group size to batch size.")
    parser.add_argument("--batch_size", type=int, default=32, help="Total number of environments to run in parallel (batch_size = group_size * (number of groups))")
    parser.add_argument("--optim_steps", type=int, default=30, help="The number of optimization steps within each annealing step")

    # Reward bump detector arguments
    parser.add_argument("--bump_threshold", type=float, default=50, help="The threshold for the reward bump detector")
    parser.add_argument("--initial_averaging_episodes", type=int, default=1, help="The number of episodes to average the reward over")
    parser.add_argument("--min_good_samples", type=int, default=2, help="The minimum number of good samples to detect a reward bump")

    # Validation arguments
    parser.add_argument("--val_freq", type=int, default=0, help="Frequency of validation in terms of annealing steps (0 to disable)")
    parser.add_argument("--val_batch", type=int, default=32, help="Number of environments to run during validation")
    parser.add_argument("--render_freq", type=int, default=0, help="Frequency of rendering in terms of annealing steps (0 to disable)")
    args = parser.parse_args()
    # fmt: on

    enable_group_initialization = not args.disable_group_initialization

    return (
        args.env_name,
        args.anneal_steps,
        args.learning_rate,
        args.temp_start,
        args.temp_end,
        args.clip_eps,
        args.group_size if enable_group_initialization else args.batch_size,
        enable_group_initialization,
        args.batch_size,
        args.optim_steps,
        args.bump_threshold,
        args.initial_averaging_episodes,
        args.min_good_samples,
        args.val_freq,
        args.val_batch,
        args.render_freq,
    )


if __name__ == "__main__":
    main(*parse_args())
