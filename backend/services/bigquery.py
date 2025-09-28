from google.cloud import bigquery
import os
import pandas as pd
from backend.services.adk import run_fraud_detection
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class BigQueryService:
    def __init__(self):
        self.client = bigquery.Client()
        self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "vaulted-timing-473322-f9")
        self.dataset_id = os.environ.get("BIGQUERY_DATASET", "ieee_cis_fraud")

    def get_invoices(self, limit=100):
        log.info("Fetching invoices from BigQuery...")
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
            invoices = df.to_dict('records')
            log.info(f"Found {len(invoices)} invoices.")
            
            for invoice in invoices:
                fraud_result = run_fraud_detection(invoice)
                invoice['confidence'] = fraud_result.get('risk_score', 0)
                invoice['status'] = 'rejected' if fraud_result.get('risk_score', 0) > 0.5 else 'approved'
                invoice['issues'] = 1 if fraud_result.get('risk_score', 0) > 0.5 else 0
                invoice['date'] = '2024-01-01'
                invoice['description'] = 'Sample Description'

            return invoices
        except Exception as e:
            log.error(f"Error fetching invoices from BigQuery: {e}")
            return []

bigquery_service = BigQueryService()
