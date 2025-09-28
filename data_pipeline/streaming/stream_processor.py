import asyncio
import json
import logging
from typing import Callable, Optional, Any, Dict, List

import pandas as pd
from datetime import datetime
from google.cloud import pubsub_v1


class StreamProcessor:
    """Process streaming data for real-time ML inference"""

    def __init__(self, project_id: str, data_pipeline: Any):
        # Use Any for data_pipeline to avoid import-time name errors in tests
        self.project_id = project_id
        self.data_pipeline = data_pipeline
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
        self.logger = logging.getLogger(__name__)

    def create_streaming_pipeline(
        self, topic_name: str, subscription_name: str, process_func: Callable
    ):
        """Create a streaming pipeline with Pub/Sub"""

        # Create topic and subscription if they don't exist
        topic_path = self.publisher.topic_path(self.project_id, topic_name)
        subscription_path = self.subscriber.subscription_path(
            self.project_id, subscription_name
        )

        try:
            self.publisher.create_topic(request={"name": topic_path})
        except Exception:
            pass  # Topic already exists

        try:
            self.subscriber.create_subscription(
                request={"name": subscription_path, "topic": topic_path}
            )
        except Exception:
            pass  # Subscription already exists

        # Stream processing callback
        def callback(message):
            try:
                # Parse message
                data = json.loads(message.data.decode("utf-8"))

                # Process data
                result = process_func(data)

                # Log result
                self.logger.info(f"Processed message: {message.message_id}")

                # Acknowledge message
                message.ack()

                # Store result if needed
                if result:
                    self._store_stream_result(result)

            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                message.nack()

        # Start streaming
        streaming_pull_future = self.subscriber.subscribe(
            subscription_path, callback=callback
        )

        return streaming_pull_future

    def _store_stream_result(self, result: Dict[str, Any]):
        """Store streaming results to BigQuery"""
        # Accept either a list of records or a single dict
        if isinstance(result, list):
            df = pd.DataFrame(result)
        else:
            df = pd.DataFrame([result])

        df["processed_at"] = datetime.now()

        table_id = f"{self.project_id}.streaming_results.predictions"
        # Use BigQuery client if available on pipeline, otherwise fallback to to_gbq
        if hasattr(self.data_pipeline, "bq_client"):
            self.data_pipeline.bq_client.load_table_from_dataframe(df, table_id)
        else:
            df.to_gbq(table_id, project_id=self.project_id, if_exists="append")

    async def process_batch_stream(self, messages: List[Dict], batch_size: int = 100):
        """Process messages in batches for efficiency"""

        for i in range(0, len(messages), batch_size):
            batch = messages[i : i + batch_size]

            # Convert to DataFrame
            df = pd.DataFrame(batch)

            # Apply feature engineering
            df = self.data_pipeline.feature_store.engineer_features(df, "transaction")

            # Run batch predictions
            # This would call your ML model
            predictions = await self._batch_predict(df)

            # Ensure predictions align with messages. If model returns a single
            # value or wrong-length list, expand or truncate to match batch.
            preds = list(predictions)
            if len(preds) == 1 and len(batch) > 1:
                preds = preds * len(batch)
            if len(preds) != len(batch):
                # Truncate or pad with None to match length
                if len(preds) > len(batch):
                    preds = preds[: len(batch)]
                else:
                    preds = preds + [None] * (len(batch) - len(preds))

            # Build results ensuring consistent column lengths
            results_df = pd.DataFrame(
                {
                    "message_id": [m.get("id") or None for m in batch],
                    "prediction": preds,
                    "timestamp": [datetime.now() for _ in batch],
                }
            )

            self._store_stream_result(results_df.to_dict("records"))
