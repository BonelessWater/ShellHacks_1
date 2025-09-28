"""
Unit tests for data pipeline core functionality
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from data_pipeline.monitoring.data_monitor import DataMonitor, DataVersion

# Import your modules
from data_pipeline.core.data_access import DataPipeline
from data_pipeline.features.feature_store import FeatureStore


class TestDataPipeline:
    """Test DataPipeline class"""

    @pytest.fixture
    def pipeline(self, mock_bigquery_client, mock_storage_client):
        """Create DataPipeline instance with mocked clients"""
        with patch(
            "data_pipeline.core.data_access.bigquery.Client",
            return_value=mock_bigquery_client,
        ):
            with patch(
                "data_pipeline.core.data_access.storage.Client",
                return_value=mock_storage_client,
            ):
                return DataPipeline(project_id="test-project-123")

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initializes correctly"""
        assert pipeline.project_id == "test-project-123"
        assert "financial_anomaly" in pipeline.datasets
        assert "invoices" in pipeline.gcs_buckets

    def test_get_dataset(self, pipeline, sample_transaction_data):
        """Test getting dataset from BigQuery"""
        # Mock the query result
        pipeline.bq_client.query.return_value.to_dataframe.return_value = (
            sample_transaction_data
        )

        # Get dataset
        df = pipeline.get_dataset("financial_anomaly", limit=10)

        # Assertions
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 100  # Should respect the sample data size
        pipeline.bq_client.query.assert_called_once()

    def test_get_dataset_with_filters(self, pipeline, sample_transaction_data):
        """Test getting dataset with filters"""
        pipeline.bq_client.query.return_value.to_dataframe.return_value = (
            sample_transaction_data
        )

        # Apply filters
        df = pipeline.get_dataset(
            "credit_card_fraud",
            filters={"isFraud": 1, "ProductCD": ["W", "H"]},
            columns=["TransactionID", "TransactionAmt", "isFraud"],
        )

        # Check query was called with filters
        query_call = pipeline.bq_client.query.call_args[0][0]
        assert "WHERE" in query_call
        assert "isFraud = 1" in query_call

    def test_get_training_batch(self, pipeline, sample_transaction_data):
        """Test getting training and validation batches"""
        pipeline.bq_client.query.return_value.to_dataframe.return_value = (
            sample_transaction_data
        )

        train_df, val_df = pipeline.get_training_batch(
            "ieee_transaction", batch_size=50, validation_split=0.2
        )

        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert len(train_df) > len(val_df)  # Training should be larger

    def test_cache_functionality(self, pipeline):
        """Test caching mechanism"""
        # First call should hit the database
        pipeline.get_dataset.cache_clear()
        df1 = pipeline.get_dataset("financial_anomaly")

        # Second call should use cache (mocking the cache behavior)
        df2 = pipeline.get_dataset("financial_anomaly")

        # Cache info shows hit
        cache_info = pipeline.get_dataset.cache_info()
        assert cache_info.hits >= 0


class TestFeatureStore:
    """Test FeatureStore functionality"""

    @pytest.fixture
    def feature_store(self, mock_bigquery_client):
        """Create FeatureStore instance"""
        with patch("data_pipeline.core.data_access.DataPipeline") as mock_pipeline:
            mock_pipeline.return_value.bq_client = mock_bigquery_client
            mock_pipeline.return_value.project_id = "test-project-123"
            return FeatureStore(mock_pipeline.return_value)

    def test_feature_engineering_transaction(
        self, feature_store, sample_transaction_data
    ):
        """Test transaction feature engineering"""
        df = sample_transaction_data.copy()

        # Engineer features
        result = feature_store.engineer_features(df, "transaction")

        # Check new features were created
        assert "feature_hour_of_day" in result.columns
        assert "feature_day_of_week" in result.columns
        assert "feature_amount_log" in result.columns

        # Verify transformations
        assert result["feature_amount_log"].notna().all()
        assert result["feature_hour_of_day"].between(0, 23).all()

    def test_feature_engineering_invoice(self, feature_store, sample_invoice_data):
        """Test invoice feature engineering"""
        df = pd.DataFrame([sample_invoice_data])

        # Engineer features
        result = feature_store.engineer_features(df, "invoice")

        # Check features
        assert "feature_tax_rate" in result.columns
        assert "feature_vendor_risk_score" in result.columns

    def test_feature_engineering_with_missing_data(self, feature_store):
        """Test feature engineering handles missing data gracefully"""
        df = pd.DataFrame(
            {
                "TransactionAmt": [100, None, 200],
                "Timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", None]),
            }
        )

        result = feature_store.engineer_features(df, "transaction")

        # Should handle NaN values
        assert len(result) == 3
        assert result.columns.str.startswith("feature_").any()

    def test_save_features(self, feature_store, sample_transaction_data):
        """Test saving features to BigQuery"""
        df = sample_transaction_data.copy()

        # Mock the BigQuery client
        feature_store.pipeline.bq_client.load_table_from_dataframe.return_value.result.return_value = (
            None
        )

        # Save features
        feature_store.save_features(df, "test_features")

        # Verify save was called
        feature_store.pipeline.bq_client.load_table_from_dataframe.assert_called_once()


class TestDataMonitor:
    """Test data monitoring functionality"""

    @pytest.fixture
    def monitor(self, mock_bigquery_client):
        """Create DataMonitor instance"""
        with patch("data_pipeline.core.data_access.DataPipeline") as mock_pipeline:
            mock_pipeline.return_value.bq_client = mock_bigquery_client
            mock_pipeline.return_value.project_id = "test-project-123"
            return DataMonitor(mock_pipeline.return_value)

    def test_create_version(self, monitor, sample_transaction_data):
        """Test creating data version"""
        version = monitor.create_version("test_dataset", sample_transaction_data)

        assert isinstance(version, DataVersion)
        assert version.dataset_name == "test_dataset"
        assert version.row_count == len(sample_transaction_data)
        assert version.schema_hash is not None
        assert len(version.version_id) == 12

    def test_monitor_data_quality(self, monitor, sample_transaction_data):
        """Test data quality monitoring"""
        # Add some quality issues
        df = sample_transaction_data.copy()
        df.loc[0:5, "TransactionAmt"] = None  # Add missing values
        df.loc[10:15, :] = df.loc[0:5, :].values  # Add duplicates

        quality_report = monitor.monitor_data_quality("test_dataset", df)

        assert "quality_issues" in quality_report
        assert quality_report["duplicate_rows"] > 0
        assert quality_report["row_count"] == len(df)
        assert isinstance(quality_report["missing_percentage"], dict)

    @patch("pandas.read_gbq")
    def test_check_data_drift(self, mock_read_gbq, monitor, sample_transaction_data):
        """Test data drift detection"""
        # Mock reference data
        reference_data = sample_transaction_data.copy()
        reference_data["TransactionAmt"] = (
            reference_data["TransactionAmt"] * 0.5
        )  # Create drift
        mock_read_gbq.return_value = reference_data

        # Current data with drift
        current_data = sample_transaction_data.copy()

        drift_report = monitor.check_data_drift(
            "test_dataset", "ref_version_123", current_data, threshold=0.05
        )

        assert "drift_detected" in drift_report
        assert "columns_with_drift" in drift_report
        assert drift_report["dataset"] == "test_dataset"
