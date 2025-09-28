"""Example CLI to run an Optuna study.

This script is intentionally guarded: it will refuse to run unless the
environment variable RUN_OPTUNA=1 is set or the --confirm flag is passed.
It also demonstrates how to log trials to a JSONL file for auditability.
"""
import argparse
import os
import json
from backend.ml import optuna_utils


def dummy_objective(trial):
    # A trivial objective for demonstration: tune a number to get close to 0.5
    x = trial.suggest_float("x", 0.0, 1.0)
    return (x - 0.5) ** 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--log", type=str, default="optuna_trials.jsonl")
    parser.add_argument("--confirm", action="store_true", help="Confirm you want to run Optuna locally")
    args = parser.parse_args()

    # If not explicitly confirmed via flag, require the env var
    require_opt_in = not args.confirm

    try:
        result = optuna_utils.run_optuna_study(dummy_objective, n_trials=args.trials, trial_log_path=args.log, require_opt_in=require_opt_in)
        print("Study finished:", result)
    except Exception as e:
        print("Refusing to run Optuna study:", e)


if __name__ == "__main__":
    main()
