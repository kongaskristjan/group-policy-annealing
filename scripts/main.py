from argparse import ArgumentParser

from lib.grouped_environments import GroupedEnvironments


def main(env_name: str, steps: int, learning_rate: float, temperature: float, group_size: int, batch_size: int, num_iters: int) -> None:
    envs = GroupedEnvironments(env_name, group_size, batch_size)
    obs = envs.reset()

    print(envs.num_observations)
    print(envs.num_actions)


def parse_args() -> tuple[str, int, float, float, int, int, int]:
    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole-v1")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_iters", type=int, default=1)
    args = parser.parse_args()
    return args.env_name, args.steps, args.learning_rate, args.temperature, args.group_size, args.batch_size, args.num_iters


if __name__ == "__main__":
    main(*parse_args())
