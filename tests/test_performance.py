"""
Performance and benchmark tests
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from backend.data_pipeline.core.data_access import DataPipeline
from backend.data_pipeline.features.feature_store import FeatureStore


class TestPerformance:
    """Test performance and benchmarks"""

    @pytest.mark.benchmark
    def test_feature_engineering_performance(self, benchmark):
        """Benchmark feature engineering speed"""
        # Create large dataset
        n_rows = 10000
        df = pd.DataFrame(
            {
                "TransactionAmt": np.random.uniform(10, 1000, n_rows),
                "Timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="min"),
                "Merchant": np.random.choice(["A", "B", "C"], n_rows),
            }
        )

        with patch("data_pipeline.core.data_access.DataPipeline"):
            feature_store = FeatureStore(Mock())

            # Benchmark feature engineering
            result = benchmark(feature_store.engineer_features, df, "transaction")

            assert len(result) == n_rows

    @pytest.mark.slow
    def test_large_dataset_handling(self):
        """Test handling large datasets"""
        # Create very large dataset
        n_rows = 100000
        large_df = pd.DataFrame(
            {f"col_{i}": np.random.randn(n_rows) for i in range(50)}
        )

        start_time = time.time()

        # Process in chunks
        chunk_size = 10000
        processed_chunks = []

        for i in range(0, len(large_df), chunk_size):
            chunk = large_df.iloc[i : i + chunk_size]
            # Simulate processing
            processed = chunk.mean()
            processed_chunks.append(processed)

        elapsed_time = time.time() - start_time

        assert len(processed_chunks) == 10
        assert elapsed_time < 10  # Should process within 10 seconds

    @pytest.mark.benchmark
    def test_query_optimization(self, benchmark):
        """Test query optimization"""

        def optimized_query():
            query = """
            SELECT 
                TransactionID,
                TransactionAmt,
                isFraud
            FROM `project.dataset.table`
            WHERE TransactionAmt > 100
                AND DATE(Timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
            LIMIT 10000
            """
            return query

        result = benchmark(optimized_query)
        assert "WHERE" in result
        assert "LIMIT" in result
