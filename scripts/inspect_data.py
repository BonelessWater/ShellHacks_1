import os
from google.cloud import bigquery

def inspect_data():
    """
    Connects to BigQuery and prints the first 5 rows of the sample_transactions table.
    """
    client = bigquery.Client()

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "vaulted-timing-473322-f9")
    dataset_id = os.environ.get("BIGQUERY_DATASET", "ieee_cis_fraud")
    table_id = f"{project_id}.{dataset_id}.sample_transactions"

    print(f"Reading data from {table_id}...")
    sql = f"SELECT * FROM `{table_id}` LIMIT 5"
    
    try:
        df = client.query(sql).to_dataframe(create_bqstorage_client=False)
        print(df)
    except Exception as e:
        print(f"Error fetching data from BigQuery: {e}")

if __name__ == "__main__":
    inspect_data()
