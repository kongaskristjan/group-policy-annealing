from argparse import ArgumentParser

import torch

from lib.anneal import anneal_batch_episode
from lib.grouped_environments import GroupedEnvironments
from lib.model import get_model
from lib.sample import render_episode, sample_batch_episode, validate


def main(
    env_name: str,
    anneal_steps: int,
    learning_rate: float,
    temperature: float,
    group_size: int,
    batch_size: int,
    optim_steps: int,
    val_freq: int,
    val_batch: int,
    render_freq: int,
) -> None:

    envs = GroupedEnvironments(env_name, group_size, batch_size)
    model = get_model(envs.num_observations, envs.num_actions, hidden=[32])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for step in range(anneal_steps):
        if render_freq > 0 and step % render_freq == 0:
            render_episode(model, env_name)

        if val_freq > 0 and step % val_freq == 0:
            val_reward = validate(model, env_name, val_batch)
            print(f"Validation [{step} ({step * batch_size} samples)]: mean_reward - {val_reward:.2f}")

        observations, actions, rewards, done_mask = sample_batch_episode(model, envs)
        loss = anneal_batch_episode(model, observations, actions, rewards, done_mask, optimizer, temperature, group_size, optim_steps)[0]
        print(f"Annealing [{step}/{anneal_steps} ({step * batch_size} samples)]: mean_reward - {torch.mean(rewards)}, loss - {loss}")


def parse_args() -> tuple[str, int, float, float, int, int, int, int, int, int]:
    # fmt: off
    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole-v1", help="The name of the environment to run")
    parser.add_argument("--anneal_steps", type=int, default=100, help="The number of annealings to run (not related to environment steps)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="The learning rate for the optimizer")
    parser.add_argument("--temperature", type=float, default=0.5, help="The temperature of the annealing algorithm")
    parser.add_argument("--group_size", type=int, default=8, help="The number of environments in each group with identical environment seeds")
    parser.add_argument("--batch_size", type=int, default=32, help="Total number of environments to run in parallel (batch_size = group_size * (number of groups))")
    parser.add_argument("--optim_steps", type=int, default=30, help="The number of optimization steps within each annealing step")

    # Validation arguments
    parser.add_argument("--val_freq", type=int, default=0, help="Frequency of validation in terms of annealing steps (0 to disable)")
    parser.add_argument("--val_batch", type=int, default=32, help="Number of environments to run during validation")
    parser.add_argument("--render_freq", type=int, default=0, help="Frequency of rendering in terms of annealing steps (0 to disable)")
    args = parser.parse_args()
    # fmt: on

    return (
        args.env_name,
        args.anneal_steps,
        args.learning_rate,
        args.temperature,
        args.group_size,
        args.batch_size,
        args.optim_steps,
        args.val_freq,
        args.val_batch,
        args.render_freq,
    )


if __name__ == "__main__":
    main(*parse_args())
