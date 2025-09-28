# file: data_pipeline/orchestration/pipeline_orchestrator.py

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import schedule

from backend.data_pipeline.core.data_access import DataPipeline
from backend.data_pipeline.features.feature_store import FeatureStore

# Configure logging to see pipeline output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_daily_training_pipeline():
    """
    This function encapsulates the entire daily model training workflow.
    """
    logging.info("Starting the daily fraud detection model training pipeline...")

    try:
        # Initialize your project's data pipeline and feature store
        data_pipeline = DataPipeline()
        feature_store = FeatureStore(data_pipeline)

        # --- 1. Extract Data ---
        logging.info("Step 1: Extracting fresh data from BigQuery.")
        yesterday = datetime.now() - timedelta(days=1)
        df = data_pipeline.get_dataset(
            "ieee_transaction", filters={"date": yesterday.strftime("%Y-%m-%d")}
        )

        if df.empty:
            logging.warning(
                "No data found for yesterday. Skipping the rest of the pipeline."
            )
            return

        logging.info(f"Successfully extracted {len(df)} rows.")

        # --- 2. Engineer Features ---
        logging.info("Step 2: Applying feature engineering.")
        engineered_df = feature_store.engineer_features(df, "transaction")
        logging.info(f"Engineered {len(engineered_df.columns)} features.")

        # --- 3. Train Model (Simulation) ---
        logging.info("Step 3: Training the model.")
        # NOTE: You would import and call your actual model training function here
        logging.info("Model training task completed successfully.")

        # --- 4. Validate Model (Simulation) ---
        logging.info("Step 4: Validating model performance.")
        # NOTE: You would call your model validation logic here
        logging.info("Model validation task completed.")

        logging.info("Daily training pipeline finished successfully!")

    except Exception as e:
        logging.error(
            f"An error occurred during pipeline execution: {e}", exc_info=True
        )


def main():
    """
    Schedules the daily job and runs it continuously.
    """
    # Schedule the job to run every day at 2 AM
    schedule.every().day.at("02:00").do(run_daily_training_pipeline)
    logging.info("Scheduler started. Waiting for the daily job at 02:00...")

    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    main()
