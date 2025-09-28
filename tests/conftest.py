"""
Pytest configuration and fixtures for all tests
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from faker import Faker

# Set test environment
os.environ["TESTING"] = "true"

# Prefer an explicitly set GOOGLE_APPLICATION_CREDENTIALS. If not set,
# fall back to a local path used by the developer on this machine.
default_creds = (
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    or "/Users/ilandanial/Downloads/vaulted-timing-473322-f9-5f312b3321cc.json"
)
if os.path.exists(default_creds):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = default_creds

# If the credentials JSON contains a project_id, set GOOGLE_CLOUD_PROJECT so
# BigQuery clients pick up the right project by default.
try:
    import json as _json

    with open(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")) as _f:
        _j = _json.load(_f)
        if _j.get("project_id"):
            os.environ["GOOGLE_CLOUD_PROJECT"] = _j.get("project_id")
except Exception:
    # Best-effort only; leave env as-is if reading fails.
    pass

# Initialize faker for test data
fake = Faker()

# Default dataset for tests when only a table name is provided
os.environ.setdefault('TEST_BQ_DATASET', 'transactional_fraud')
os.environ.setdefault('BQ_DEFAULT_DATASET', os.environ.get('TEST_BQ_DATASET'))


@pytest.fixture(scope="session")
def test_project_id():
    """Test project ID"""
    # Return configured project if available so tests use the real BigQuery project
    return os.getenv("GOOGLE_CLOUD_PROJECT", "test-project-123")


@pytest.fixture
def mock_bigquery_client():
    """Mock BigQuery client"""
    with patch("google.cloud.bigquery.Client") as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        # Mock query results
        mock_result = MagicMock()
        mock_result.to_dataframe.return_value = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3],
                "TransactionAmt": [100.0, 200.0, 150.0],
                "isFraud": [0, 1, 0],
            }
        )

        client_instance.query.return_value.result.return_value = mock_result

        yield client_instance


@pytest.fixture
def mock_storage_client():
    """Mock Google Cloud Storage client"""
    with patch("google.cloud.storage.Client") as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        # Mock bucket and blob
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.return_value = b"fake_image_data"
        mock_bucket.blob.return_value = mock_blob
        client_instance.bucket.return_value = mock_bucket

        yield client_instance


@pytest.fixture
def sample_transaction_data():
    """Generate sample transaction data"""
    n_samples = 100
    return pd.DataFrame(
        {
            "TransactionID": range(1, n_samples + 1),
            "TransactionAmt": np.random.uniform(10, 1000, n_samples),
            "ProductCD": np.random.choice(["W", "H", "C", "S", "R"], n_samples),
            "card1": np.random.randint(1000, 9999, n_samples),
            "card2": np.random.randint(100, 999, n_samples),
            "card3": np.random.randint(100, 999, n_samples),
            "card4": np.random.choice(["discover", "mastercard", "visa"], n_samples),
            "card5": np.random.randint(100, 999, n_samples),
            "card6": np.random.choice(["debit", "credit"], n_samples),
            "TransactionDT": np.random.randint(10000000, 20000000, n_samples),
            "isFraud": np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            "Timestamp": pd.date_range(start="2024-01-01", periods=n_samples, freq="h"),
        }
    )


@pytest.fixture
def sample_invoice_data():
    """Generate sample invoice data"""
    return {
        "invoice_number": fake.uuid4(),
        "vendor": {"name": fake.company(), "address": fake.address(), "verified": True},
        # Faker's Generator may not have date_past/date_future on all versions;
        # use date_between which is broadly available.
        "invoice_date": fake.date_between(start_date='-30d', end_date='today').isoformat(),
        "due_date": fake.date_between(start_date='today', end_date='+30d').isoformat(),
        "subtotal": 1000.00,
        "tax_amount": 100.00,
        "total_amount": 1100.00,
        "line_items": [
            {"description": fake.word(), "quantity": 2, "price": 500.00}
        ],
    }


@pytest.fixture
def sample_image_metadata():
    """Generate sample image metadata for forgery detection"""
    return pd.DataFrame(
        {
            "image_gcs_path": [f"gs://test-bucket/image_{i}.jpg" for i in range(10)],
            "is_fraudulent": np.random.choice([True, False], 10),
            "class_name": np.random.choice(["invoice", "receipt", "document"], 10),
            "x_center_norm": np.random.uniform(0, 1, 10),
            "y_center_norm": np.random.uniform(0, 1, 10),
            "width_norm": np.random.uniform(0.1, 0.5, 10),
            "height_norm": np.random.uniform(0.1, 0.5, 10),
        }
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_api_keys():
    """Mock API keys for testing"""
    return ["test_api_key_1", "test_api_key_2", "test_api_key_3"]


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Setup test environment variables"""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project-123")


@pytest.fixture(autouse=True)
def prevent_bq_writes(monkeypatch):
    """Prevent accidental writes to BigQuery during tests.

    By default this fixture monkeypatches common write paths to be no-ops.
    To allow real writes (dangerous), set ALLOW_REAL_BQ_WRITES=true in the
    environment before running pytest.
    """
    allow = os.getenv("ALLOW_REAL_BQ_WRITES", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    if allow:
        yield
        return

    # No-op pandas.DataFrame.to_gbq
    try:
        import pandas as _pd

        def _noop_to_gbq(self, *args, **kwargs):
            # Return without writing; mimic successful write when tests expect it
            return None

        monkeypatch.setattr(_pd.DataFrame, "to_gbq", _noop_to_gbq, raising=False)
    except Exception:
        pass

    # No-op google.cloud.bigquery.Client.load_table_from_dataframe
    try:
        from google.cloud import bigquery as _bq

        def _noop_load_table_from_dataframe(self, df, table_id, job_config=None):
            class _Job:
                def result(self):
                    return None

            return _Job()

        monkeypatch.setattr(
            _bq.Client, "load_table_from_dataframe", _noop_load_table_from_dataframe, raising=False
        )
    except Exception:
        pass

    # No-op client.insert_rows_from_dataframe (legacy method used in some modules)
    try:
        from google.cloud import bigquery as _bq2

        def _noop_insert_rows_from_dataframe(self, table_id, df, *args, **kwargs):
            # Return empty error list to indicate success
            return []

        monkeypatch.setattr(
            _bq2.Client, "insert_rows_from_dataframe", _noop_insert_rows_from_dataframe, raising=False
        )
    except Exception:
        pass

    yield
