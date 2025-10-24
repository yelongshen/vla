"""Evaluate a 'gr00t' model checkpoint in the lerobot pybullet stairs environment.

Usage:
  python examples/eval_gr00t.py --ckpt /path/to/checkpoint.pt --factory mypkg.models.build_model

The factory (optional) is a dotted path to a model constructor that accepts a config dict and returns an nn.Module.
"""
import argparse
from lerobot.evaluator import evaluate_on_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=False)
    parser.add_argument('--factory', required=False, help='dotted path to model factory')
    parser.add_argument('--episodes', type=int, default=5)
    args = parser.parse_args()

    res = evaluate_on_env('pybullet', ckpt_path=args.ckpt, factory=args.factory, episodes=args.episodes)
    print('Evaluation result:', res)


if __name__ == '__main__':
    main()
