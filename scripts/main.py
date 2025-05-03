from argparse import ArgumentParser

import numpy as np

from lib.anneal import anneal_batch_episode
from lib.grouped_environments import GroupedEnvironments
from lib.model import get_model, sample_batch_episode


def main(env_name: str, anneal_steps: int, learning_rate: float, temperature: float, group_size: int, batch_size: int, optim_steps: int) -> None:
    envs = GroupedEnvironments(env_name, group_size, batch_size)
    model = get_model(envs.num_observations, envs.num_actions, hidden=[])

    for _ in range(anneal_steps):
        actions, probs, rewards, done_mask = sample_batch_episode(model, envs)
        loss = anneal_batch_episode(model, actions, probs, rewards, done_mask, learning_rate, temperature, optim_steps)
        print(f"Annealing step {_}/{anneal_steps}: avg_reward - {np.mean(rewards)}, loss - {loss}")


def parse_args() -> tuple[str, int, float, float, int, int, int]:
    # fmt: off
    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole-v1", help="The name of the environment to run")
    parser.add_argument("--anneal_steps", type=int, default=100, help="The number of annealings to run (not related to environment steps)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="The learning rate for the optimizer")
    parser.add_argument("--temperature", type=float, default=1.0, help="The temperature of the annealing algorithm")
    parser.add_argument("--group_size", type=int, default=8, help="The number of environments in each group with identical environment seeds")
    parser.add_argument("--batch_size", type=int, default=32, help="Total number of environments to run in parallel (batch_size = group_size * (number of groups))")
    parser.add_argument("--optim_steps", type=int, default=1, help="The number of optimization steps within each annealing step")
    args = parser.parse_args()
    # fmt: on

    return args.env_name, args.anneal_steps, args.learning_rate, args.temperature, args.group_size, args.batch_size, args.optim_steps


if __name__ == "__main__":
    main(*parse_args())
