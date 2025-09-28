import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class DataVersion:
    """Data version metadata"""

    version_id: str
    dataset_name: str
    created_at: datetime
    row_count: int
    schema_hash: str
    statistics: Dict[str, Any]


class DataMonitor:
    """Monitor data quality and track versions"""

    def __init__(self, data_pipeline: Any):
        # Use Any to avoid import-time forward references to DataPipeline
        self.pipeline = data_pipeline
        self.monitoring_table = f"{self.pipeline.project_id}.data_monitoring.metrics"
        self.version_table = f"{self.pipeline.project_id}.data_monitoring.versions"

    def create_version(self, dataset_name: str, df: pd.DataFrame) -> DataVersion:
        """Create a new version of the dataset"""

        # Generate version ID
        timestamp = datetime.now()
        version_id = hashlib.md5(
            f"{dataset_name}_{timestamp.isoformat()}".encode()
        ).hexdigest()[:12]

        # Calculate statistics
        statistics = {
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(
                include=["object"]
            ).columns.tolist(),
            "missing_values": df.isnull().sum().to_dict(),
            "unique_counts": {col: df[col].nunique() for col in df.columns},
            "basic_stats": df.describe().to_dict() if not df.empty else {},
        }

        # Create schema hash
        schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        schema_hash = hashlib.md5(
            json.dumps(schema, sort_keys=True).encode()
        ).hexdigest()

        version = DataVersion(
            version_id=version_id,
            dataset_name=dataset_name,
            created_at=timestamp,
            row_count=len(df),
            schema_hash=schema_hash,
            statistics=statistics,
        )

        # Save version metadata
        self._save_version(version)

        # Save actual data with version
        versioned_table = (
            f"{self.pipeline.project_id}.versioned_data.{dataset_name}_{version_id}"
        )
        df.to_gbq(
            versioned_table, project_id=self.pipeline.project_id, if_exists="replace"
        )

        return version

    def _save_version(self, version: DataVersion):
        """Save version metadata to BigQuery"""
        version_df = pd.DataFrame([asdict(version)])
        version_df.to_gbq(
            self.version_table, project_id=self.pipeline.project_id, if_exists="append"
        )

    def check_data_drift(
        self,
        dataset_name: str,
        reference_version: str,
        current_data: pd.DataFrame,
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """Check for data drift between versions"""

        # Load reference data
        ref_table = f"{self.pipeline.project_id}.versioned_data.{dataset_name}_{reference_version}"
        ref_data = pd.read_gbq(ref_table, project_id=self.pipeline.project_id)

        drift_report = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "reference_version": reference_version,
            "drift_detected": False,
            "columns_with_drift": [],
        }

        # Check numeric columns for distribution shift
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in ref_data.columns:
                # Kolmogorov-Smirnov test
                from scipy import stats

                ks_stat, p_value = stats.ks_2samp(
                    ref_data[col].dropna(), current_data[col].dropna()
                )

                if p_value < threshold:
                    drift_report["columns_with_drift"].append(
                        {
                            "column": col,
                            "ks_statistic": ks_stat,
                            "p_value": p_value,
                            "reference_mean": ref_data[col].mean(),
                            "current_mean": current_data[col].mean(),
                        }
                    )
                    drift_report["drift_detected"] = True

        # Log drift report
        self._log_monitoring_event("data_drift", drift_report)

        return drift_report

    def monitor_data_quality(
        self, dataset_name: str, df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Monitor data quality metrics"""

        quality_metrics = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "row_count": len(df),
            "duplicate_rows": df.duplicated().sum(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "quality_issues": [],
        }

        # Check for quality issues
        if quality_metrics["duplicate_rows"] > 0:
            quality_metrics["quality_issues"].append(
                f"Found {quality_metrics['duplicate_rows']} duplicate rows"
            )

        for col, missing_pct in quality_metrics["missing_percentage"].items():
            if missing_pct > 20:
                quality_metrics["quality_issues"].append(
                    f"Column {col} has {missing_pct:.1f}% missing values"
                )

        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 3 * IQR) | (df[col] > Q3 + 3 * IQR)).sum()
            if outliers > 0:
                quality_metrics["quality_issues"].append(
                    f"Column {col} has {outliers} extreme outliers"
                )

        # Log metrics
        self._log_monitoring_event("quality_check", quality_metrics)

        return quality_metrics

    def _log_monitoring_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log monitoring events to BigQuery"""
        event_df = pd.DataFrame(
            [
                {
                    "event_type": event_type,
                    "timestamp": datetime.now(),
                    "event_data": json.dumps(event_data),
                }
            ]
        )

        event_df.to_gbq(
            self.monitoring_table,
            project_id=self.pipeline.project_id,
            if_exists="append",
        )
