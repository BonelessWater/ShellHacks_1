import os

from dotenv import load_dotenv
from google.cloud import bigquery
from google.api_core import exceptions as google_exceptions
from google.oauth2 import service_account

load_dotenv()


class BigQueryManager:
    """Simple wrapper around google.cloud.bigquery.Client with clear availability
    reporting. This avoids raising during import time when credentials are not
    present and gives the rest of the codebase a predictable `is_available()`
    check.
    """
    def __init__(self):
        self.project_id = os.getenv("BQ_PROJECT_ID", "vaulted-timing-473322-f9")
        self.client = None
        self.available = False
        self.init_error = None

        # Try to initialize a client using explicit service account file when
        # provided via GOOGLE_APPLICATION_CREDENTIALS, otherwise rely on the
        # default application credentials available in the environment.
        try:
            key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if key_path:
                # If path exists, create credentials explicitly to avoid
                # surprising behavior when the path is invalid.
                if os.path.exists(key_path):
                    creds = service_account.Credentials.from_service_account_file(key_path)
                    self.client = bigquery.Client(credentials=creds, project=self.project_id)
                    self.available = True
                else:
                    # Invalid path - fall back to trying ADC but mark init_error
                    self.init_error = f"GOOGLE_APPLICATION_CREDENTIALS points to missing file: {key_path}"
                    try:
                        self.client = bigquery.Client(project=self.project_id)
                        self.available = True
                    except Exception as e:
                        self.client = None
                        self.init_error = f"Failed to initialize default client after bad key path: {e}"
            else:
                # No explicit key file - try default credentials
                try:
                    self.client = bigquery.Client(project=self.project_id)
                    self.available = True
                except Exception as e:
                    self.client = None
                    self.init_error = str(e)
        except Exception as e:
            self.client = None
            self.available = False
            self.init_error = str(e)

        # Define your datasets
        self.datasets = {
            "fraud": f"{self.project_id}.transactional_fraud",
            "ieee": f"{self.project_id}.ieee_cis_fraud",
            "documents": f"{self.project_id}.document_forgery",
        }

    def is_available(self):
        return bool(self.available and self.client is not None)

    def query(self, sql_query):
        """Execute a BigQuery SQL query"""
        if not self.is_available():
            raise RuntimeError(f"BigQuery client is not available: {self.init_error}")
        query_job = self.client.query(sql_query)
        try:
            # Prefer fast path that may use the BigQuery Storage API
            return query_job.result().to_dataframe()
        except Exception as e:
            # If the execution failed due to missing BigQuery Storage API
            # permission (common for restricted service accounts), retry
            # without creating a bqstorage client which avoids that permission.
            try:
                if isinstance(e, google_exceptions.PermissionDenied) or (
                    isinstance(e, Exception) and 'bigquery.readsessions.create' in str(e)
                ):
                    return query_job.result().to_dataframe(create_bqstorage_client=False)
            except Exception:
                # Fall through to raise the original exception below
                pass
            # If the retry didn't succeed, raise so callers can handle the error
            raise

    def load_data_to_bq(self, dataframe, dataset_id, table_id):
        """Load a pandas DataFrame to BigQuery"""
        if not self.is_available():
            raise RuntimeError(f"BigQuery client is not available: {self.init_error}")

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
try:
    bq_manager = BigQueryManager()
except Exception as e:
    # Ensure import-time failures don't crash the application. Other modules
    # will check `bq_manager.is_available()` before using it.
    bq_manager = BigQueryManager()
