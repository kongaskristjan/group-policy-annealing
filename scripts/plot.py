from argparse import ArgumentParser, Namespace
from pathlib import Path

from lib.tracking import load_run, plot_percentiles


def main(args: Namespace) -> None:
    percentiles = [float(p) for p in args.percentiles.split(",")]
    runs = [load_run(file) for file in args.files]
    print(f"Plotting a total of {len(runs)} experiment runs:")
    for i, run in enumerate(runs):
        print(f"Experiment {i + 1}:")
        for key, value in run.parameters.items():
            print(f"  - {key}: {value}")
        print()
    plot_percentiles(runs, percentiles)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Plot reward percentiles from experiment JSON files")
    parser.add_argument("files", nargs="+", type=Path, help="Paths to experiment JSON files")
    parser.add_argument("--percentiles", type=str, default="20,50,80", help="Comma-separated percentiles to plot (default: 20,50,80)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
