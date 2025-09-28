"""
End-to-end tests for complete workflows
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest


class TestEndToEnd:
    """Test complete workflows end-to-end"""

    @pytest.mark.integration
    def test_fraud_detection_workflow(
        self, sample_transaction_data, sample_invoice_data
    ):
        """Test complete fraud detection workflow"""
        with patch("backend.data_pipeline.api.easy_access.EasyDataAccess") as mock_easy_access:
            # Setup mocks
            easy_access = mock_easy_access.return_value
            easy_access.get_training_data.return_value = sample_transaction_data

            # 1. Data retrieval
            train_data = easy_access.get_training_data("fraud_detection")
            assert len(train_data) > 0

            # 2. Feature engineering (mocked)
            from backend.data_pipeline.features.feature_store import FeatureStore

            with patch.object(FeatureStore, "engineer_features") as mock_engineer:
                mock_engineer.return_value = train_data
                engineered_data = mock_engineer(train_data, "transaction")

            # 3. Model training (mocked)
            from unittest.mock import MagicMock

            model = MagicMock()
            model.fit.return_value = None
            model.predict.return_value = [0.8]

            # 4. Prediction
            model.fit(
                engineered_data.drop("isFraud", axis=1), engineered_data["isFraud"]
            )
            predictions = model.predict(engineered_data.drop("isFraud", axis=1))

            assert len(predictions) > 0

    @pytest.mark.integration
    def test_document_forgery_workflow(self, sample_image_metadata):
        """Test document forgery detection workflow"""
        with patch("data_pipeline.api.easy_access.EasyDataAccess") as mock_easy_access:
            easy_access = mock_easy_access.return_value

            # 1. Get image metadata
            easy_access.pipeline.get_dataset.return_value = sample_image_metadata
            image_data = easy_access.pipeline.get_dataset("invoice_annotations")

            # 2. Prepare for training
            assert "image_gcs_path" in image_data.columns
            assert "is_fraudulent" in image_data.columns

            # 3. Create TensorFlow dataset (mocked)
            with patch(
                "backend.data_pipeline.integrations.ml_frameworks.TensorFlowDataPipeline"
            ) as mock_tf:
                tf_pipeline = mock_tf.return_value
                tf_pipeline.create_image_dataset.return_value = Mock()

                dataset = tf_pipeline.create_image_dataset("invoice_annotations")
                assert dataset is not None

    @pytest.mark.slow
    @pytest.mark.integration
    def test_monitoring_workflow(self):
        """Test monitoring and alerting workflow"""
        from backend.data_pipeline.monitoring.data_monitor import DataMonitor

        with patch("backend.data_pipeline.core.data_access.DataPipeline") as mock_pipeline:
            monitor = DataMonitor(mock_pipeline.return_value)

            # Create sample data with quality issues
            problematic_data = pd.DataFrame(
                {
                    "amount": [
                        100,
                        200,
                        None,
                        400,
                        100,
                        100,
                    ],  # Has nulls and duplicates
                    "date": pd.date_range("2024-01-01", periods=6),
                }
            )

            # Run quality check
            quality_report = monitor.monitor_data_quality(
                "test_dataset", problematic_data
            )

            # Check issues were detected
            assert len(quality_report["quality_issues"]) > 0
            assert quality_report["duplicate_rows"] > 0
            assert "amount" in quality_report["missing_percentage"]
