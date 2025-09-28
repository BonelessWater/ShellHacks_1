"""Feature store shim that re-exports the implementation from core.data_access

The real FeatureStore implementation lives in `data_pipeline.core.data_access`.
This shim keeps the old import path used by tests and other modules.
"""

from data_pipeline.core.data_access import FeatureStore

__all__ = ["FeatureStore"]
