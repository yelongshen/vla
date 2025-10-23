"""Command-line interface placeholder for lerobot.

Subcommands will be implemented in later steps.
"""
import argparse


def build_parser():
    parser = argparse.ArgumentParser(prog="lerobot", description="LeRobot CLI")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--config", "-c", required=True, help="Path to config YAML")

    p_eval = sub.add_parser("eval", help="Evaluate a model checkpoint")
    p_eval.add_argument("--ckpt", required=True, help="Path to checkpoint file")
    p_eval.add_argument("--config", "-c", required=False, help="Path to config YAML (optional if included in checkpoint)")

    p_deploy = sub.add_parser("deploy", help="Run model deployment server")
    p_deploy.add_argument("--ckpt", required=True, help="Path to checkpoint file")
    p_deploy.add_argument("--host", default="127.0.0.1", help="Host to bind the server")
    p_deploy.add_argument("--port", type=int, default=8000, help="Port for the server")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "train":
        from .trainer import train
        train(args.config)
    elif args.command == "eval":
        from .evaluate import evaluate
        evaluate(args.ckpt, args.config)
    elif args.command == "deploy":
        # import here to avoid heavy dependencies during train/eval runs
        try:
            from .deploy import serve
        except Exception as e:
            print("Deployment module not yet implemented or failed to import:", e)
            return 2
        serve(args.ckpt, host=args.host, port=args.port)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
