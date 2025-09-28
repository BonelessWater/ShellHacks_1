import os
from google.cloud import bigquery
import pandas as pd
import numpy as np

def create_sample_data():
    """
    Reads data from the source transaction table, randomizes it slightly,
    and uploads it to a new table called sample_transactions.
    """
    client = bigquery.Client()

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "vaulted-timing-473322-f9")
    dataset_id = os.environ.get("BIGQUERY_DATASET", "ieee_cis_fraud")
    source_table_id = f"{project_id}.{dataset_id}.train_transaction"
    destination_table_id = f"{project_id}.{dataset_id}.sample_transactions"

    print(f"Reading data from {source_table_id}...")
    sql = f"SELECT * FROM `{source_table_id}` LIMIT 1000"
    
    # Disable the BigQuery Storage API to avoid permission issues
    df = client.query(sql).to_dataframe(create_bqstorage_client=False)

    print("Randomizing data...")
    # Add small random noise to TransactionAmt
    noise = np.random.normal(0, 0.05, df.shape[0])
    df['TransactionAmt'] = df['TransactionAmt'] * (1 + noise)
    df['TransactionAmt'] = df['TransactionAmt'].round(2)

    # Shuffle card types to simulate different vendors
    df['card4'] = np.random.permutation(df['card4'])

    print(f"Uploading randomized data to {destination_table_id}...")
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
    )
    job = client.load_table_from_dataframe(
        df, destination_table_id, job_config=job_config
    )
    job.result()
    print(f"Successfully created table {destination_table_id} with {len(df)} rows.")

if __name__ == "__main__":
    create_sample_data()
