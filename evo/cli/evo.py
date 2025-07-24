import sys
import argparse

def clean_command(args):
    import os
    import shutil
    import glob

    def prompt_confirm():
        warning = (
            "WARNING: This will permanently delete the following:\n"
            "- checkpoints/\n- logs/\n- trained_models/\n- all .csv files in the project\n- all __pycache__ directories\n\nContinue? [Y/n]: "
        )
        return input(warning).strip() == 'Y'

    def remove_path(path):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            os.remove(path)

    if prompt_confirm():
        # Remove directories
        for d in ["checkpoints", "logs", "trained_models", "tensorboard_logs"]:
            if os.path.exists(d):
                print(f"Removing {d}/ ...")
                remove_path(d)
        
        # Remove CSV files
        for csv in glob.glob("**/*.csv", recursive=True):
            print(f"Removing {csv} ...")
            remove_path(csv)
        
        # Remove __pycache__ directories
        for root, dirs, files in os.walk("."):
            for d in dirs:
                if d == "__pycache__":
                    pycache_path = os.path.join(root, d)
                    print(f"Removing {pycache_path} ...")
                    remove_path(pycache_path)
        print("Clean complete.")
    else:
        print("Clean Aborted.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="EVO unified CLI: data, train, backtest"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Data subcommand
    data_parser = subparsers.add_parser("data", help="Data download and processing CLI")
    data_parser.add_argument('args', nargs=argparse.REMAINDER)

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Training CLI")
    train_parser.add_argument('args', nargs=argparse.REMAINDER)

    # Backtest subcommand
    backtest_parser = subparsers.add_parser("backtest", help="Backtesting CLI")
    backtest_parser.add_argument('args', nargs=argparse.REMAINDER)

    # Genetic subcommand
    genetic_parser = subparsers.add_parser("genetic", help="Genetic optimization CLI")
    genetic_parser.add_argument('args', nargs=argparse.REMAINDER)

    # Clean subcommand
    clean_parser = subparsers.add_parser("clean", help="Remove temporary/generated files (checkpoints, logs, trained_models, CSVs, pycaches)")

    args = parser.parse_args()

    if args.command == "data":
        from .data import data_command
        data_command(args.args)
    elif args.command == "train":
        from .train import train_command
        train_command(args.args)
    elif args.command == "backtest":
        from .backtest import backtest_command
        backtest_command(args.args)
    elif args.command == "clean":
        clean_command(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 