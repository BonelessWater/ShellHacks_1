"""Example script to load data from BigQuery and train a model.

This script is intentionally guarded: training from production data and cloud
resources requires an explicit opt-in. Use RUN_TRAINING=1 or pass --confirm.
"""
import argparse
import os

from backend.ml.transaction_trainer import TransactionAnomalyTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="BigQuery SQL query to load training data")
    parser.add_argument("--table", type=str, help="BigQuery table name (project.dataset.table or dataset.table)")
    parser.add_argument("--project", type=str, help="GCP project for BigQuery")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--trials", type=int, default=0, help="Number of optuna trials (optional)")
    parser.add_argument("--confirm", action="store_true", help="Confirm you want to run training")
    parser.add_argument("--save-dir", type=str, default="./models", help="Directory to save model artifacts")
    args = parser.parse_args()

    require_opt_in = not args.confirm
    if require_opt_in and os.environ.get("RUN_TRAINING", "0") != "1":
        print("Training from BigQuery is disabled by default. Set RUN_TRAINING=1 or pass --confirm to proceed.")
        return

    trainer = TransactionAnomalyTrainer()
    res = trainer.train_on_bq(query=args.query, table=args.table, project=args.project, dataset=args.dataset, n_trials=args.trials, optuna_search=(args.trials>0), save_dir=args.save_dir, require_opt_in=False)
    print("Training finished. Result:", res)


if __name__ == "__main__":
    main()
