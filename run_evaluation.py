import argparse
from pathlib import Path
from src.evaluation.evaluator import Evaluator


def get_args():
    parser = argparse.ArgumentParser(description='Arguments to compute the mesh')
    parser.add_argument('--checkpoint_path', type=str, help='SLAM checkpoint path', default="output/slam/full_experiment/")
    parser.add_argument('--config_path', type=str, help='Config path', default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.config_path == "":
        args.config_path = Path(args.checkpoint_path) / "config.yaml"

    evaluator = Evaluator(Path(args.checkpoint_path), Path(args.config_path))
    evaluator.run()
