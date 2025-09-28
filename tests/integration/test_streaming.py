"""
Test streaming data pipeline
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from data_pipeline.streaming.stream_processor import StreamProcessor


class TestStreamingPipeline:
    """Test streaming functionality"""

    @pytest.fixture
    def stream_processor(self, mock_bigquery_client):
        """Create stream processor"""
        with patch("data_pipeline.core.data_access.DataPipeline") as mock_pipeline:
            with patch("google.cloud.pubsub_v1.PublisherClient"):
                with patch("google.cloud.pubsub_v1.SubscriberClient"):
                    pipeline = mock_pipeline.return_value
                    return StreamProcessor("test-project", pipeline)

    def test_create_streaming_pipeline(self, stream_processor):
        """Test creating streaming pipeline"""

        def process_func(data):
            return {"processed": True, "data": data}

        # Mock the subscription
        with patch.object(stream_processor.subscriber, "subscribe") as mock_subscribe:
            future = stream_processor.create_streaming_pipeline(
                "test_topic", "test_subscription", process_func
            )

            mock_subscribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_stream_processing(
        self, stream_processor, sample_transaction_data
    ):
        """Test batch stream processing"""
        messages = sample_transaction_data.head(10).to_dict("records")

        # Mock batch predict
        async def mock_predict(df):
            return [0.5] * len(df)

        stream_processor._batch_predict = mock_predict

        # Process batch
        with patch.object(stream_processor, "_store_stream_result"):
            await stream_processor.process_batch_stream(messages, batch_size=5)

            # Should be called twice (10 messages / batch_size 5)
            assert stream_processor._store_stream_result.call_count == 2
