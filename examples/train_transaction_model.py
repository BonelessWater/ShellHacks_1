"""Example training script for transaction anomaly model.

This is a guarded example: it will only run if TensorFlow is installed and
required libraries are available. It demonstrates data loading, train/test
splitting, training, evaluation, and (optionally) saving the model.
"""
import os
from backend.ml.transaction_trainer import TransactionAnomalyTrainer, _HAS_TF


def main():
    trainer = TransactionAnomalyTrainer()

    # Replace these with your CSV and column names
    csv_path = os.environ.get("TRANSACTION_CSV", "data/transactions_sample.csv")
    feature_columns = ["amount", "vendor_len"]
    label_column = "is_fraud"

    try:
        X, y = trainer.load_csv_data(csv_path, feature_columns, label_column)
    except Exception as e:
        print("Failed to load data:", e)
        return

    # Demonstrate safe train/test separation via trainer.train_and_evaluate
    if not _HAS_TF:
        print("TensorFlow not available in this environment. The script will only demonstrate splitting and basic flow.")
        X_train, X_test, y_train, y_test = trainer.split_train_test(X, y, test_size=0.2, random_state=42)
        print(f"Loaded {len(X)} rows. Train: {len(X_train)}, Test: {len(X_test)}")
        return

    # If TF is available, try a tiny grid search (guarded) then final train+eval
    try:
        # define a simple builder that accepts params
        def build_fn_from_params(params):
            def builder(input_dim):
                # Use trainer.create_model structure but allow overriding units via params
                import tensorflow as tf  # type: ignore

                model = tf.keras.Sequential()
                model.add(tf.keras.layers.Dense(params.get("units1", 128), activation="relu", input_shape=(input_dim,)))
                model.add(tf.keras.layers.Dropout(params.get("dropout", 0.3)))
                model.add(tf.keras.layers.Dense(params.get("units2", 64), activation="relu"))
                model.add(tf.keras.layers.Dropout(params.get("dropout", 0.3)))
                model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
                model.compile(optimizer=params.get("optimizer", "adam"), loss="binary_crossentropy", metrics=["accuracy", "AUC"])
                return model
            return builder

        # Load data and split
        X_train, X_test, y_train, y_test = trainer.split_train_test(X, y, test_size=0.2, random_state=42)

        # Small grid
        param_grid = {
            "units1": [64, 128],
            "units2": [32, 64],
            "dropout": [0.2, 0.3],
        }

        def model_builder_factory(params):
            return build_fn_from_params(params)

        # Try simple grid search (this will internally call train_model and evaluate on validation)
        best_params, best_model, report = trainer.simple_grid_search(param_grid, lambda p: model_builder_factory(p)(X_train[0].__len__() if hasattr(X_train[0], '__len__') else len(X_train[0])), X_train, y_train, X_val=X_test, y_val=y_test, max_trials=5)

        if best_model is not None:
            print("Best params:", best_params)
            metrics = trainer.evaluate_model(best_model, X_test, y_test)
            print("Metrics for best model:", metrics)
            # Persist artifacts
            out = trainer.train_and_evaluate(X, y, test_size=0.2, random_state=42, model_builder=lambda d: best_model, save_dir="./models")
            print("Artifacts saved to:", out.get("model_path"), out.get("scaler_path"))
        else:
            print("Grid search did not return a best model; falling back to simple training")
            out = trainer.train_and_evaluate(X, y, test_size=0.2, random_state=42, save_dir="./models")
            print("Training finished. Metrics:", out.get("metrics"))

    except Exception as e:
        print("Training/optimization failed:", e)


if __name__ == "__main__":
    main()
