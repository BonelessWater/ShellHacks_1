"""Lightweight transaction anomaly trainer helpers.

This module provides a thin wrapper around TensorFlow and Vertex AI when
available. The module is guarded so tests and local dev don't require heavy
dependencies. It also documents how to use the trainer in production.
"""
import logging
from typing import Any

log = logging.getLogger("transaction_trainer")

try:
    import tensorflow as tf  # type: ignore
    from google.cloud import aiplatform  # type: ignore

    _HAS_TF = True
except Exception:
    tf = None
    aiplatform = None
    _HAS_TF = False

# Provide a placeholder name for tests that monkeypatch backend.ml.transaction_trainer.bigquery
# Some tests do monkeypatch.setattr('backend.ml.transaction_trainer.bigquery', ...)
bigquery = None


class TransactionAnomalyTrainer:
    """Trainer wrapper. Only functional when TensorFlow and Vertex SDK are installed.

    Methods are no-ops when dependencies are missing; they log helpful messages.
    """

    def create_model(self, input_dim: int = 20):
        if not _HAS_TF:
            raise RuntimeError("TensorFlow not available in this environment")

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu", input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "AUC"])
        return model

    def train_and_deploy(self, training_data: Any, bucket: str, project: str, display_name: str = "fraud-detection-model"):
        if not _HAS_TF:
            raise RuntimeError("TensorFlow/Vertex SDK not available")

        model = self.create_model(input_dim=training_data.shape[1])
        model.fit(training_data)

        # Save to GCS and deploy to Vertex AI
        artifact_uri = f"gs://{bucket}/{display_name}"
        model.save(artifact_uri)

        aiplatform.init(project=project)
        model_res = aiplatform.Model.upload(display_name=display_name, artifact_uri=artifact_uri, serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest")
        endpoint = model_res.deploy(machine_type="n1-standard-4", min_replica_count=1, max_replica_count=3)
        return endpoint

    # ----------------- Utilities for safer training & evaluation -----------------

    def load_csv_data(self, csv_path: str, feature_columns: list, label_column: str):
        """Load CSV data into (X, y) arrays. Uses pandas if available, otherwise a lightweight CSV reader.

        Returns: tuple (X, y) where X is a 2D list/array and y is a 1D list/array.
        """
        try:
            import pandas as pd  # type: ignore

            df = pd.read_csv(csv_path)
            X = df[feature_columns].values
            y = df[label_column].values
            return X, y
        except Exception:
            # Fallback simple CSV reader (no dtype guarantees)
            import csv

            X = []
            y = []
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    X.append([float(row[c]) for c in feature_columns])
                    y.append(float(row[label_column]))
            return X, y

    def load_from_bigquery(self, query: str = None, table: str = None, project: str = None, dataset: str = None, feature_columns: list = None, label_column: str = None):
        """Load data from BigQuery into (X, y).

        Either provide a SQL `query` or `table` in the form 'dataset.table' (project optional).
        This function is guarded: if the BigQuery client isn't available it raises a
        clear ImportError. It will try to return numpy arrays when possible.
        """
        try:
            from google.cloud import bigquery  # type: ignore
        except Exception as e:  # pragma: no cover - depends on environment
            raise ImportError("google-cloud-bigquery is not installed or not available") from e

        client_kwargs = {}
        if project:
            client_kwargs["project"] = project

        client = bigquery.Client(**client_kwargs)

        if query:
            bq_query = query
        elif table:
            # if table looks like dataset.table or project.dataset.table
            if project and dataset:
                bq_query = f"SELECT * FROM `{project}.{dataset}.{table}`"
            else:
                bq_query = f"SELECT * FROM `{table}`"
        else:
            raise ValueError("Either query or table must be provided")

        # Execute query
        job = client.query(bq_query)
        rows = job.result()

        # Convert to lists
        cols = rows.schema
        # If feature_columns/label_column not provided, attempt best-effort auto-detect
        all_cols = [s.name for s in cols]
        if feature_columns is None or label_column is None:
            # Heuristic: last column is label
            if len(all_cols) < 2:
                raise ValueError("Insufficient columns in BigQuery result to derive features/label")
            feature_columns = feature_columns or all_cols[:-1]
            label_column = label_column or all_cols[-1]

        X = []
        y = []
        for r in rows:
            try:
                X.append([r[c] for c in feature_columns])
                y.append(r[label_column])
            except Exception:
                # If field access by name fails, convert to dict
                d = dict(r.items())
                X.append([d.get(c) for c in feature_columns])
                y.append(d.get(label_column))

        try:
            import numpy as _np  # type: ignore

            return _np.array(X), _np.array(y)
        except Exception:
            return X, y

    def train_on_bq(self, *, query: str = None, table: str = None, project: str = None, dataset: str = None, feature_columns: list = None, label_column: str = None, n_trials: int = 0, optuna_search: bool = False, save_dir: str = "./models", require_opt_in: bool = True):
        """High-level helper to load training data from BigQuery and train a model.

        This function performs safety checks: it requires an explicit opt-in via
        env `RUN_TRAINING=1` unless `require_opt_in=False`. It also guards imports
        so that local dev without GCP/TF SDKs won't break.
        """
        import os

        if require_opt_in and os.environ.get("RUN_TRAINING", "0") != "1":
            raise RuntimeError("Training from BigQuery is disabled by default. Set RUN_TRAINING=1 to proceed.")

        # Load data
        X_raw, y_raw = self.load_from_bigquery(query=query, table=table, project=project, dataset=dataset, feature_columns=feature_columns, label_column=label_column)

        # Optional feature transform hook (import feature_utils lazily to avoid heavy deps)
        feature_transform = None
        try:
            from backend.ml.feature_utils import rows_to_feature_matrix
            # If X_raw came back as numpy arrays of rows, try to convert to dict rows
            rows_for_transform = None
            try:
                # X_raw may be a numpy array of dict-like objects or 2D numeric array
                import numpy as _np  # type: ignore
                if hasattr(X_raw, "dtype") and X_raw.dtype == object:
                    rows_for_transform = [dict(r) for r in X_raw]
            except Exception:
                rows_for_transform = None

            if rows_for_transform:
                X, y = rows_to_feature_matrix(rows_for_transform, feature_columns=feature_columns, label_column=label_column)
            else:
                # Assume X_raw is already numeric 2D
                X, y = X_raw, y_raw
        except Exception:
            X, y = X_raw, y_raw

        # Optional Optuna tuning (thin wrapper). If optuna not available this will raise.
        if optuna_search and n_trials > 0:
            try:
                from backend.ml.optuna_utils import run_optuna_study
            except Exception:
                raise RuntimeError("Optuna is required for optuna_search but is not available")

            # Example simple objective: wrap train_and_evaluate with sampled hyperparams
            def objective(trial):
                # A few simple searchable params
                lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
                dropout = trial.suggest_float("dropout", 0.0, 0.5)

                def builder(input_dim):
                    # Use existing create_model but apply small modifications
                    if not _HAS_TF:
                        raise RuntimeError("TF not available for builder")
                    import tensorflow as _tf  # type: ignore

                    model = _tf.keras.Sequential([
                        _tf.keras.layers.Dense(128, activation="relu", input_shape=(input_dim,)),
                        _tf.keras.layers.Dropout(dropout),
                        _tf.keras.layers.Dense(64, activation="relu"),
                        _tf.keras.layers.Dropout(dropout),
                        _tf.keras.layers.Dense(32, activation="relu"),
                        _tf.keras.layers.Dense(1, activation="sigmoid"),
                    ])
                    model.compile(optimizer=_tf.keras.optimizers.Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"])
                    return model

                # Train with a single small epoch to get a proxy metric
                try:
                    res = self.train_and_evaluate(X, y, test_size=0.2, model_builder=builder, save_dir=save_dir)
                    # Optuna minimizes — return negative accuracy if accuracy available
                    acc = res.get("metrics", {}).get("accuracy")
                    if acc is None:
                        return 1.0
                    return 1.0 - float(acc)
                except Exception:
                    return 1.0

            # Run optuna study (requires opt-in inside run_optuna_study as well)
            run_optuna_study(objective, n_trials=n_trials, trial_log_path=os.path.join(save_dir, "optuna_trials.jsonl"))

        # Finally, run full training & evaluation
        res = self.train_and_evaluate(X, y, save_dir=save_dir)

        # Write a simple manifest.json with artifact locations and dataset fingerprint
        try:
            from backend.ml.manifest import write_manifest

            manifest = {
                "model_path": res.get("model_path"),
                "scaler_path": res.get("scaler_path"),
                "metadata_path": res.get("metadata_path"),
            }
            # Attempt to read dataset fingerprint from metadata
            try:
                import json as _json
                if res.get("metadata_path"):
                    with open(res["metadata_path"], "r") as fh:
                        md = _json.load(fh)
                        manifest["dataset_fingerprint"] = md.get("dataset_fingerprint")
                        manifest["metrics"] = md.get("metrics")
            except Exception:
                pass

            write_manifest(os.path.join(save_dir, "manifest.json"), manifest)
            # Attempt to register model in a lightweight registry (best-effort)
            try:
                from backend.ml.registry import register_model

                registry_path = os.environ.get("MODEL_REGISTRY_PATH", os.path.join(save_dir, "model_registry.json"))
                entry = {
                    "manifest": manifest,
                    "saved_at": os.path.join(save_dir, "manifest.json"),
                }
                try:
                    register_model(registry_path, entry)
                except Exception:
                    # don't fail training because registry write failed
                    log.warning("Failed to register model in registry")
            except Exception:
                # registry module may not be present; ignore
                pass
        except Exception:
            pass

        return res

    def split_train_test(self, X, y, test_size: float = 0.2, random_state: int | None = None):
        """Split dataset into train/test. Uses sklearn if available, otherwise falls back to a simple numpy-based shuffle.

        Returns: (X_train, X_test, y_train, y_test)
        """
        try:
            from sklearn.model_selection import train_test_split  # type: ignore

            return train_test_split(X, y, test_size=test_size, random_state=random_state)
        except Exception:
            try:
                import numpy as _np  # type: ignore

                X_arr = _np.array(X)
                y_arr = _np.array(y)
                rng = _np.random.RandomState(random_state) if random_state is not None else _np.random
                idx = rng.permutation(len(X_arr))
                cutoff = int(len(X_arr) * (1 - test_size))
                train_idx = idx[:cutoff]
                test_idx = idx[cutoff:]
                return X_arr[train_idx], X_arr[test_idx], y_arr[train_idx], y_arr[test_idx]
            except Exception:
                # Last resort: deterministic split
                n = len(X)
                cutoff = int(n * (1 - test_size))
                return X[:cutoff], X[cutoff:], y[:cutoff], y[cutoff:]

    def evaluate_model(self, model, X_test, y_test) -> dict:
        """Evaluate a trained model and return metrics dict. Tries to use sklearn metrics when available.

        For TF models, model.predict(X_test) is used.
        """
        metrics = {}
        try:
            import numpy as _np  # type: ignore

            y_pred_probs = None
            if _HAS_TF and hasattr(model, "predict"):
                y_pred_probs = model.predict(X_test)
                # flatten
                y_pred_probs = _np.array(y_pred_probs).reshape(-1)
            else:
                # If model is a callable predictor
                y_pred_probs = _np.array(model(X_test))

            # Threshold for binary
            y_pred = (y_pred_probs >= 0.5).astype(float)

            try:
                from sklearn.metrics import accuracy_score, roc_auc_score  # type: ignore

                metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
                # roc_auc may fail if only one class present
                try:
                    metrics["auc"] = float(roc_auc_score(y_test, y_pred_probs))
                except Exception:
                    metrics["auc"] = None
            except Exception:
                # Fallback simple accuracy
                correct = float((y_pred == _np.array(y_test)).sum())
                metrics["accuracy"] = correct / len(y_test)
                metrics["auc"] = None

            return metrics
        except Exception as e:
            log.warning(f"Evaluation failed: {e}")
            return {"error": str(e)}

    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs: int = 10, batch_size: int = 32, model_builder: Any = None, verbose: int = 1):
        """Train a model using TF if available. model_builder is a callable that returns a compiled TF model given input_dim.

        Returns the trained model.
        """
        if not _HAS_TF:
            raise RuntimeError("TensorFlow not available in this environment")

        import numpy as _np  # type: ignore

        X_train = _np.array(X_train)
        y_train = _np.array(y_train)

        input_dim = X_train.shape[1]
        builder = model_builder or (lambda dim: self.create_model(input_dim=dim))
        model = builder(input_dim)

        callbacks = []
        try:
            from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

            callbacks.append(EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True))
        except Exception:
            pass

        if X_val is not None and y_val is not None:
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)
        else:
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)

        return model, history

    def simple_grid_search(self, param_grid: dict, build_fn, X_train, y_train, X_val=None, y_val=None, max_trials: int = 10):
        """Very small grid-search style loop that trains multiple models and returns best by validation accuracy.

        param_grid: dict of param -> list(values)
        build_fn: callable(params) -> compiled model
        Returns: best_params, best_model, report_list
        """
        # Build cartesian product (but limited)
        import itertools

        keys = list(param_grid.keys())
        combos = list(itertools.product(*(param_grid[k] for k in keys)))
        combos = combos[:max_trials]
        best = None
        report = []
        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                model = build_fn(params)
                trained, _ = self.train_model(X_train, y_train, X_val=X_val, y_val=y_val, model_builder=lambda d: model, epochs=5)
                metrics = self.evaluate_model(trained, X_val if X_val is not None else X_train, y_val if y_val is not None else y_train)
                acc = metrics.get("accuracy") or 0.0
                report.append({"params": params, "metrics": metrics})
                if best is None or acc > best[0]:
                    best = (acc, params, trained)
            except Exception as e:
                report.append({"params": params, "error": str(e)})

        if best:
            return best[1], best[2], report
        return None, None, report

    def train_and_evaluate(self, X, y, feature_scaler=None, test_size: float = 0.2, random_state: int = 42, model_builder: Any = None, save_dir: str = "./models"):
        """High-level wrapper: split data, fit scaler, train model, evaluate, and persist artifacts.

        Returns dict with keys: model, metrics, scaler_path, model_path.
        """
        # Enforce separation
        X_train, X_test, y_train, y_test = self.split_train_test(X, y, test_size=test_size, random_state=random_state)

        # Preprocessing
        from backend.ml.model_utils import SimpleScaler, save_preprocessor

        scaler = feature_scaler or SimpleScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model, history = self.train_model(X_train_scaled, y_train, X_val=X_test_scaled, y_val=y_test, model_builder=model_builder)

        # Evaluate
        metrics = self.evaluate_model(model, X_test_scaled, y_test)

        # Persist
        import os
        import json
        os.makedirs(save_dir, exist_ok=True)
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        model_path = os.path.join(save_dir, "model")
        metadata_path = os.path.join(save_dir, "metadata.json")
        try:
            save_preprocessor(scaler, scaler_path)
        except Exception as e:
            log.warning(f"Failed to save scaler: {e}")

        try:
            if _HAS_TF and hasattr(model, "save"):
                model.save(model_path)
        except Exception as e:
            log.warning(f"Failed to save model: {e}")

        # Compute a dataset fingerprint for reproducibility and save as metadata
        try:
            from backend.ml.dataset_utils import fingerprint_rows

            # Fingerprint training rows by combining features + label per row
            combined = []
            for xi, yi in zip(X_train, y_train):
                # ensure iteration works for numpy arrays or lists
                combined.append(list(xi) + [yi])
            dataset_fp = fingerprint_rows(combined)
        except Exception as e:
            log.warning(f"Failed to compute dataset fingerprint: {e}")
            dataset_fp = None

        try:
            meta = {"metrics": metrics, "dataset_fingerprint": dataset_fp}
            with open(metadata_path, "w") as fh:
                json.dump(meta, fh)
        except Exception as e:
            log.warning(f"Failed to write metadata: {e}")

        return {"model": model, "metrics": metrics, "scaler_path": scaler_path, "model_path": model_path, "metadata_path": metadata_path}
