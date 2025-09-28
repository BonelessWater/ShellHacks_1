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
                COALESCE(P_emaildomain, 'Unknown Vendor') as vendor
            FROM `{table_id}`
            LIMIT {limit}
        """
        try:
            df = self.client.query(query).to_dataframe(create_bqstorage_client=False)
            invoices = df.to_dict('records')
            log.info(f"Found {len(invoices)} invoices.")
            
            for invoice in invoices:
                fraud_result = run_fraud_detection(invoice)
                risk_score = fraud_result.get('risk_score', 0)
                invoice['confidence'] = risk_score
                
                if risk_score >= 0.75:
                    invoice['status'] = 'approved'
                    invoice['riskLevel'] = 'Low'
                    invoice['issues'] = 0
                elif risk_score > 0.5 and risk_score < 0.75:
                    invoice['status'] = 'review_required'
                    invoice['riskLevel'] = 'Medium'
                    invoice['issues'] = 1
                else:
                    invoice['status'] = 'rejected'
                    invoice['riskLevel'] = 'High'
                    invoice['issues'] = 3

                invoice['date'] = '2024-01-01'
                invoice['description'] = 'Sample Description'

            return invoices
        except Exception as e:
            log.error(f"Error fetching invoices from BigQuery: {e}")
            return []

    def get_transactions_for_analytics(self):
        log.info("Fetching transactions for analytics...")
        table_id = f"{self.project_id}.{self.dataset_id}.sample_transactions"
        query = f"""
            SELECT 
                TransactionID,
                TransactionAmt,
                TransactionDT,
                ProductCD,
                COALESCE(P_emaildomain, 'Unknown Vendor') as vendor
            FROM `{table_id}`
        """
        try:
            df = self.client.query(query).to_dataframe(create_bqstorage_client=False)
            return df.to_dict('records')
        except Exception as e:
            log.error(f"Error fetching transactions for analytics from BigQuery: {e}")
            return []

bigquery_service = BigQueryService()
