from google.cloud import bigquery
import os
import pandas as pd

class BigQueryService:
    def __init__(self):
        self.client = bigquery.Client()
        self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "vaulted-timing-473322-f9")
        self.dataset_id = os.environ.get("BIGQUERY_DATASET", "ieee_cis_fraud")

    def get_invoices(self, limit=100):
        table_id = f"{self.project_id}.{self.dataset_id}.sample_transactions"
        query = f"""
            SELECT 
                TransactionID as id,
                TransactionAmt as amount,
                ProductCD as product_code,
                isFraud,
                COALESCE(P_emaildomain, 'Unknown Vendor') as vendor,
                CASE
                    WHEN isFraud = 1 THEN 'rejected'
                    ELSE 'approved'
                END as status,
                0.9 as confidence,
                0 as issues,
                '2024-01-01' as date,
                'Sample Description' as description
            FROM `{table_id}`
            LIMIT {limit}
        """
        try:
            df = self.client.query(query).to_dataframe(create_bqstorage_client=False)
            return df.to_dict('records')
        except Exception as e:
            print(f"Error fetching invoices from BigQuery: {e}")
            return []

bigquery_service = BigQueryService()
