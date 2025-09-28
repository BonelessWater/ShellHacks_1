import os

from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account

load_dotenv()


class BigQueryManager:
    def __init__(self):
        self.project_id = "vaulted-timing-473322-f9"

        # Initialize client with credentials
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            self.client = bigquery.Client()
        else:
            # Fallback to default credentials
            self.client = bigquery.Client(project=self.project_id)

        # Define your datasets
        self.datasets = {
            "fraud": f"{self.project_id}.transactional_fraud",
            "ieee": f"{self.project_id}.ieee_cis_fraud",
            "documents": f"{self.project_id}.document_forgery",
        }

    def query(self, sql_query):
        """Execute a BigQuery SQL query"""
        query_job = self.client.query(sql_query)
        return query_job.result().to_dataframe()

    def load_data_to_bq(self, dataframe, dataset_id, table_id):
        """Load a pandas DataFrame to BigQuery"""
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"

        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE", autodetect=True  # or WRITE_APPEND
        )

        job = self.client.load_table_from_dataframe(
            dataframe, table_ref, job_config=job_config
        )
        job.result()  # Wait for job to complete
        print(f"Loaded {len(dataframe)} rows to {table_ref}")


# Initialize the manager
bq_manager = BigQueryManager()
