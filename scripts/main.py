from argparse import ArgumentParser

import torch

from lib.anneal import anneal_batch_episode
from lib.grouped_environments import GroupedEnvironments
from lib.model import get_model, sample_batch_episode


def main(env_name: str, anneal_steps: int, learning_rate: float, temperature: float, group_size: int, batch_size: int, optim_steps: int) -> None:
    envs = GroupedEnvironments(env_name, group_size, batch_size)
    model = get_model(envs.num_observations, envs.num_actions, hidden=[32])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(anneal_steps):
        observations, actions, rewards, done_mask = sample_batch_episode(model, envs)
        loss = anneal_batch_episode(model, observations, actions, rewards, done_mask, optimizer, temperature, group_size, optim_steps)[0]
        print(f"Annealing step {_}/{anneal_steps} (total iterations - {_ * batch_size}): avg_reward - {torch.mean(rewards)}, loss - {loss}")


def parse_args() -> tuple[str, int, float, float, int, int, int]:
    # fmt: off
    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole-v1", help="The name of the environment to run")
    parser.add_argument("--anneal_steps", type=int, default=100, help="The number of annealings to run (not related to environment steps)")
    parser.add_argument("--learning_rate", type=float, default=0.02, help="The learning rate for the optimizer")
    parser.add_argument("--temperature", type=float, default=0.5, help="The temperature of the annealing algorithm")
    parser.add_argument("--group_size", type=int, default=8, help="The number of environments in each group with identical environment seeds")
    parser.add_argument("--batch_size", type=int, default=32, help="Total number of environments to run in parallel (batch_size = group_size * (number of groups))")
    parser.add_argument("--optim_steps", type=int, default=30, help="The number of optimization steps within each annealing step")
    args = parser.parse_args()
    # fmt: on

    return args.env_name, args.anneal_steps, args.learning_rate, args.temperature, args.group_size, args.batch_size, args.optim_steps


if __name__ == "__main__":
    main(*parse_args())
