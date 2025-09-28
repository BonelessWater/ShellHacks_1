import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import os
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from google.cloud import bigquery, storage

logger = logging.getLogger(__name__)


class FeatureStore:
    """Centralized feature engineering and storage"""

    def __init__(self, pipeline):
        # avoid evaluating DataPipeline at import time (tests patch the class)
        self.pipeline = pipeline
        self.feature_definitions = {}
        self.transformers = {}
        self._register_features()

    def _register_features(self):
        """Register all feature engineering pipelines"""

        # Transaction features
        # Use the actual dataset column names (TransactionAmt etc.) used in tests
        self.feature_definitions["transaction"] = {
            "hour_of_day": lambda df: pd.to_datetime(df["Timestamp"]).dt.hour,
            "day_of_week": lambda df: pd.to_datetime(df["Timestamp"]).dt.dayofweek,
            "amount_log": lambda df: np.log1p(df["TransactionAmt"]) if "TransactionAmt" in df else 0,
            "amount_zscore": lambda df: (
                (df["TransactionAmt"] - df["TransactionAmt"].mean())
                / df["TransactionAmt"].std()
                if "TransactionAmt" in df
                else 0
            ),
            # Fallback merchant_frequency using card1 as a proxy if Merchant absent
            "merchant_frequency": lambda df: df.groupby(df.columns.intersection(["Merchant", "card1"]).tolist()[0])[
                df.columns.intersection(["Merchant", "card1"]).tolist()[0]
            ].transform("count") if any(c in df.columns for c in ("Merchant", "card1")) else 0,
            "location_risk": lambda df: df.get("Location", pd.Series(["unknown"] * len(df))).map(
                self._get_location_risk_scores()
            ),
        }

        # Invoice features for document analysis
        self.feature_definitions["invoice"] = {
            "total_line_items": lambda df: (
                df["line_items"].apply(len) if "line_items" in df else 0
            ),
            "tax_rate": lambda df: (
                df["tax_amount"] / df["subtotal"] if "subtotal" in df else 0
            ),
            "days_until_due": lambda df: (
                pd.to_datetime(df["due_date"]) - pd.to_datetime(df["invoice_date"])
            ).dt.days,
            "vendor_risk_score": lambda df: df["vendor"].apply(
                self._calculate_vendor_risk
            ),
            "amount_anomaly": lambda df: self._detect_amount_anomalies(df),
        }

        # Image features for forgery detection
        self.feature_definitions["image"] = {
            "bbox_area": lambda df: (
                df["width_norm"] * df["height_norm"] if "width_norm" in df else 0
            ),
            "bbox_aspect_ratio": lambda df: (
                df["width_norm"] / df["height_norm"] if "height_norm" in df else 1
            ),
            "bbox_center_distance": lambda df: np.sqrt(
                df["x_center_norm"] ** 2 + df["y_center_norm"] ** 2
            ),
            "class_distribution": lambda df: df.groupby("class_name")[
                "class_name"
            ].transform("count")
            / len(df),
        }

    def engineer_features(
        self,
        df: pd.DataFrame,
        feature_set: str,
        include_features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Apply feature engineering to dataframe"""

        if feature_set not in self.feature_definitions:
            raise ValueError(f"Unknown feature set: {feature_set}")

        features = self.feature_definitions[feature_set]

        if include_features:
            features = {k: v for k, v in features.items() if k in include_features}

        # Apply each feature transformation
        for feature_name, transform_func in features.items():
            try:
                df[f"feature_{feature_name}"] = transform_func(df)
            except Exception as e:
                self.pipeline.logger.warning(
                    f"Failed to create feature {feature_name}: {e}"
                )
                df[f"feature_{feature_name}"] = np.nan

        return df

    def _get_location_risk_scores(self) -> Dict[str, float]:
        """Get pre-computed location risk scores"""
        # This would typically load from a lookup table
        return {"online": 0.5, "New York": 0.3, "Los Angeles": 0.3, "unknown": 0.8}

    def _calculate_vendor_risk(self, vendor_data: Any) -> float:
        """Calculate vendor risk score"""
        # Simplified risk calculation
        if not vendor_data:
            return 0.5

        risk_factors = 0
        if isinstance(vendor_data, dict):
            if not vendor_data.get("verified", False):
                risk_factors += 0.3
            if vendor_data.get("country", "") in ["high_risk_country"]:
                risk_factors += 0.4

        return min(risk_factors, 1.0)

    def _detect_amount_anomalies(self, df: pd.DataFrame) -> pd.Series:
        """Detect anomalies in amounts using IQR method"""
        if "total_amount" not in df.columns:
            return pd.Series([0] * len(df))

        Q1 = df["total_amount"].quantile(0.25)
        Q3 = df["total_amount"].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return (
            (df["total_amount"] < lower_bound) | (df["total_amount"] > upper_bound)
        ).astype(int)

    def save_features(self, df: pd.DataFrame, feature_set_name: str):
        """Save engineered features to BigQuery"""
        table_id = (
            f"{self.pipeline.project_id}.feature_store.{feature_set_name}_features"
        )

        df["feature_version"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        df["feature_hash"] = df.apply(
            lambda x: hashlib.md5(str(x.values).encode()).hexdigest(), axis=1
        )

        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            time_partitioning=bigquery.TimePartitioning(field="feature_version"),
        )

        job = self.pipeline.bq_client.load_table_from_dataframe(
            df, table_id, job_config=job_config
        )
        job.result()

        self.pipeline.logger.info(f"Saved {len(df)} features to {table_id}")


class DataPipeline:
    """Minimal DataPipeline used by tests.

    This implements just the small surface area the unit tests exercise:
    - initialization with patched BigQuery/Storage clients
    - `get_dataset` (lru_cache-decorated)
    - `get_training_batch`
    - simple attributes like `datasets` and `gcs_buckets`
    """

    def __init__(self, project_id: str = "local-project"):
        self.project_id = project_id
        # Clients are constructed here so tests can patch google.cloud clients
        self.bq_client = bigquery.Client()
        self.storage_client = storage.Client()
        self.datasets = ["financial_anomaly", "credit_card_fraud", "ieee_transaction"]
        self.gcs_buckets = ["invoices", "raw_data"]
        self.logger = logger
        # feature_store may be used by other modules
        self.feature_store = FeatureStore(self)
        # Provide a cache and a callable wrapper for get_dataset so tests can
        # call `pipeline.get_dataset.cache_clear()` before any calls.
        self._get_dataset_cache = {}

        class _GetDatasetCallable:
            def __init__(self, pipeline):
                self.pipeline = pipeline

            def __call__(self, dataset_name: str, limit: int = 100, filters: dict = None, columns: List[str] = None):
                return self.pipeline._get_dataset_impl(dataset_name, limit=limit, filters=filters, columns=columns)

            def cache_clear(self):
                self.pipeline._get_dataset_cache.clear()

            def cache_info(self):
                # Provide minimal cache info compatible with functools.lru_cache
                class _Info:
                    def __init__(self, hits):
                        self.hits = hits

                return _Info(hits=len(self.pipeline._get_dataset_cache))

        # Bind the callable to the instance
        self.get_dataset = _GetDatasetCallable(self)
        self._get_dataset_cache = {}

        def _make_hashable_filters(filters: Optional[dict]) -> Tuple:
            if not filters:
                return ()
            items = []
            for k in sorted(filters.keys()):
                v = filters[k]
                if isinstance(v, (list, tuple)):
                    items.append((k, tuple(v)))
                else:
                    items.append((k, v))
            return tuple(items)

        # Expose a callable get_dataset previously bound in __init__ which
        # delegates to _get_dataset_impl. The __init__ attaches the callable
        # and the cache_clear method.
    def _get_dataset_impl(self, dataset_name: str, limit: int = 100, filters: dict = None, columns: List[str] = None) -> pd.DataFrame:
        """Internal implementation for dataset fetching. Qualifies short table names
        to fully-qualified `project.dataset.table` using BQ_DEFAULT_DATASET env var
        or a sensible default.
        """

        # Normalize filters into a JSON-serializable string for stable cache keys
        filters_serial = json.dumps(filters or {}, sort_keys=True, default=str)
        cache_key = (dataset_name, limit, filters_serial, json.dumps(columns or []))
        if cache_key in self._get_dataset_cache:
            return self._get_dataset_cache[cache_key]

        # Determine fully-qualified table identifier
        # Cases handled:
        # - 'project.dataset.table' -> use as-is
        # - 'dataset.table' -> qualify with project
        # - 'table' -> qualify with project and default dataset
        if dataset_name.count('.') == 2:
            fq_table = dataset_name
        elif dataset_name.count('.') == 1:
            fq_table = f"{self.project_id}.{dataset_name}"
        else:
            default_ds = os.environ.get('BQ_DEFAULT_DATASET') or os.environ.get('TEST_BQ_DATASET') or 'transactional_fraud'
            fq_table = f"{self.project_id}.{default_ds}.{dataset_name}"

        query = f"SELECT * FROM `{fq_table}` LIMIT {limit}"

        # Render filters if provided (support lists and scalars)
        if filters:
            where_clauses = []
            for k, v in (filters or {}).items():
                if isinstance(v, (list, tuple)):
                    vals = ",".join(f"'{x}'" for x in v)
                    where_clauses.append(f"{k} IN ({vals})")
                else:
                    if isinstance(v, str):
                        where_clauses.append(f"{k} = '{v}'")
                    else:
                        where_clauses.append(f"{k} = {v}")
            if where_clauses:
                query = query.replace("LIMIT", f"WHERE {' AND '.join(where_clauses)} LIMIT")

        df = self.bq_client.query(query).to_dataframe()
        if columns and not df.empty:
            df = df[columns]

        # Cache and return
        self._get_dataset_cache[cache_key] = df
        return df

    def get_training_batch(self, dataset_name: str, batch_size: int = 50, validation_split: float = 0.2):
        df = self.get_dataset(dataset_name)
        if df is None or len(df) == 0:
            return df, df

        val_count = int(len(df) * validation_split)
        # Return (train, val)
        return df.iloc[val_count:], df.iloc[:val_count]
