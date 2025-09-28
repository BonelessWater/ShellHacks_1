"""
Test ML framework integrations
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import torch

from data_pipeline.integrations.ml_frameworks import (PyTorchDataPipeline,
                                                      TensorFlowDataPipeline)


class TestTensorFlowIntegration:
    """Test TensorFlow data pipeline"""

    @pytest.fixture
    def tf_pipeline(self, mock_bigquery_client):
        """Create TensorFlow pipeline"""
        with patch("data_pipeline.core.data_access.DataPipeline") as mock_pipeline:
            with patch(
                "data_pipeline.features.feature_store.FeatureStore"
            ) as mock_feature_store:
                pipeline = mock_pipeline.return_value
                feature_store = mock_feature_store.return_value
                return TensorFlowDataPipeline(pipeline, feature_store)

    def test_create_tf_dataset(self, tf_pipeline, sample_transaction_data):
        """Test creating TensorFlow dataset"""
        # Mock data retrieval
        tf_pipeline.pipeline.get_dataset.return_value = sample_transaction_data
        tf_pipeline.feature_store.engineer_features.return_value = (
            sample_transaction_data
        )

        # Create dataset
        dataset = tf_pipeline.create_tf_dataset(
            dataset_name="test_dataset", batch_size=32, label_column="isFraud"
        )

        assert isinstance(dataset, tf.data.Dataset)

        # Test dataset produces batches
        for batch_features, batch_labels in dataset.take(1):
            assert batch_features.shape[0] <= 32
            assert batch_labels.shape[0] <= 32

    @patch("tensorflow.io.read_file")
    @patch("tensorflow.image.decode_jpeg")
    def test_create_image_dataset(
        self, mock_decode, mock_read, tf_pipeline, sample_image_metadata
    ):
        """Test creating image dataset"""
        # Mock image loading
        mock_read.return_value = tf.constant(b"fake_image_data")
        mock_decode.return_value = tf.ones((224, 224, 3))

        tf_pipeline.pipeline.get_dataset.return_value = sample_image_metadata

        # Create image dataset
        dataset = tf_pipeline.create_image_dataset(
            dataset_name="image_dataset", image_size=(224, 224), batch_size=4
        )

        assert isinstance(dataset, tf.data.Dataset)


class TestPyTorchIntegration:
    """Test PyTorch data pipeline"""

    @pytest.fixture
    def pytorch_pipeline(self, mock_bigquery_client, mock_storage_client):
        """Create PyTorch pipeline"""
        with patch("data_pipeline.core.data_access.DataPipeline") as mock_pipeline:
            with patch(
                "data_pipeline.features.feature_store.FeatureStore"
            ) as mock_feature_store:
                pipeline = mock_pipeline.return_value
                pipeline.storage_client = mock_storage_client
                feature_store = mock_feature_store.return_value
                return PyTorchDataPipeline(pipeline, feature_store)

    def test_fraud_dataset(self, pytorch_pipeline, sample_transaction_data):
        """Test PyTorch fraud dataset"""
        dataset = pytorch_pipeline.FraudDataset(sample_transaction_data, "isFraud")

        assert len(dataset) == len(sample_transaction_data)

        # Test getting items
        features, label = dataset[0]
        assert isinstance(features, torch.Tensor)
        assert isinstance(label, torch.Tensor)

    def test_create_dataloader(self, pytorch_pipeline, sample_transaction_data):
        """Test creating PyTorch DataLoader"""
        # Mock data retrieval
        pytorch_pipeline.pipeline.get_dataset.return_value = sample_transaction_data
        pytorch_pipeline.feature_store.engineer_features.return_value = (
            sample_transaction_data
        )

        # Create dataloader
        dataloader = pytorch_pipeline.create_dataloader(
            dataset_name="test_dataset",
            batch_size=16,
            shuffle=True,
            num_workers=0,  # Use 0 for testing
        )

        assert isinstance(dataloader, torch.utils.data.DataLoader)

        # Test iteration
        for batch in dataloader:
            if len(batch) == 2:  # Has labels
                features, labels = batch
                assert features.shape[0] <= 16
                assert labels.shape[0] <= 16
            break
