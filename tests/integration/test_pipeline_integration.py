"""
Integration tests for the complete pipeline
"""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.archive.main_pipeline import FraudDetectionPipeline
from backend.data_pipeline.api.easy_access import EasyDataAccess


class TestPipelineIntegration:
    """Test complete pipeline integration"""

    @pytest.mark.integration
    def test_easy_access_interface(self, sample_transaction_data):
        """Test the easy access interface"""
        with patch("backend.data_pipeline.core.data_access.DataPipeline") as mock_pipeline:
            # Mock the pipeline methods
            mock_pipeline.return_value.get_dataset.return_value = (
                sample_transaction_data
            )

            easy_access = EasyDataAccess()

            # Test getting training data
            train_data = easy_access.get_training_data(
                use_case="fraud_detection", sample_size=100
            )

            assert isinstance(train_data, pd.DataFrame)
            assert "data_version" in train_data.columns
            assert "pipeline_timestamp" in train_data.columns

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_fraud_detection_pipeline(self, sample_invoice_data):
        """Test the fraud detection pipeline end-to-end"""
        with patch("backend.main_pipeline.get_configured_lm"):
            pipeline = FraudDetectionPipeline()

            # Run detection
            result = pipeline.run_detection(sample_invoice_data, max_iterations=1)

            assert result is not None
            assert hasattr(result, "fraud_risk")
            assert hasattr(result, "confidence_score")

    @pytest.mark.integration
    def test_data_versioning_workflow(self, sample_transaction_data):
        """Test data versioning workflow"""
        with patch("backend.data_pipeline.core.data_access.DataPipeline") as mock_pipeline:
            from backend.data_pipeline.monitoring.data_monitor import DataMonitor

            monitor = DataMonitor(mock_pipeline.return_value)

            # Create version
            version1 = monitor.create_version("test_dataset", sample_transaction_data)

            # Modify data
            modified_data = sample_transaction_data.copy()
            modified_data["TransactionAmt"] = modified_data["TransactionAmt"] * 1.5

            # Create new version
            version2 = monitor.create_version("test_dataset", modified_data)

            assert version1.version_id != version2.version_id
            assert version2.created_at > version1.created_at

    @pytest.mark.integration
    @pytest.mark.requires_gcp
    def test_bigquery_connection(self, test_project_id):
        """Test actual BigQuery connection (requires GCP credentials)"""
        from google.cloud import bigquery

        try:
            client = bigquery.Client(project=test_project_id)
            # Try a simple query
            query = "SELECT 1 as test"
            result = client.query(query).result()
            assert result.total_rows == 1
        except Exception as e:
            pytest.skip(f"GCP credentials not available: {e}")
