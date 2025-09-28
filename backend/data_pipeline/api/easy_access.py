"""Easy access helpers for common data pipeline operations used in tests."""

from datetime import datetime
from typing import Optional

import pandas as pd

import backend.data_pipeline.core.data_access as data_access


class EasyDataAccess:
	"""Small helper wrapping DataPipeline to return ready-to-use dataframes."""

	def __init__(self, project_id: Optional[str] = None):
		# Import DataPipeline from module at runtime so tests can patch
		# data_pipeline.core.data_access.DataPipeline and intercept construction.
		self.pipeline = (
			data_access.DataPipeline(project_id=project_id)
			if project_id
			else data_access.DataPipeline()
		)

	def get_training_data(self, use_case: str = "fraud_detection", sample_size: int = 100) -> pd.DataFrame:
		dataset_name = "financial_anomaly" if use_case == "fraud_detection" else use_case
		df = self.pipeline.get_dataset(dataset_name, limit=sample_size)

		# Add metadata columns used by tests
		df = df.copy()
		df["data_version"] = datetime.now().strftime("%Y%m%d")
		df["pipeline_timestamp"] = datetime.now().isoformat()

		return df

__all__ = ["EasyDataAccess"]
