#!/usr/bin/env python3
"""
Google Cloud Platform Data Connector
Connects to GCP services using Application Credentials and retrieves training data
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_gcp_credentials():
    """Load Google Cloud Platform credentials"""
    load_dotenv('.env')
    
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_path:
        logger.error("GOOGLE_APPLICATION_CREDENTIALS not found in environment")
        return None
    
    if not os.path.exists(credentials_path):
        logger.error(f"Credentials file not found: {credentials_path}")
        return None
    
    logger.info(f"Found GCP credentials at: {credentials_path}")
    return credentials_path

def inspect_gcp_credentials(credentials_path):
    """Inspect the GCP credentials to understand what services are available"""
    try:
        with open(credentials_path, 'r') as f:
            creds = json.load(f)
        
        print("🔍 GCP CREDENTIALS ANALYSIS")
        print("=" * 40)
        print(f"Project ID: {creds.get('project_id', 'Not found')}")
        print(f"Client Email: {creds.get('client_email', 'Not found')}")
        print(f"Type: {creds.get('type', 'Not found')}")
        print(f"Auth URI: {creds.get('auth_uri', 'Not found')}")
        
        # Check if this is a service account
        if creds.get('type') == 'service_account':
            print("✅ Service Account detected")
            print("Available services likely include:")
            print("  - BigQuery (data warehousing)")
            print("  - Cloud Storage (file storage)")
            print("  - Firestore (document database)")
            print("  - Cloud SQL (relational database)")
            print("  - Vertex AI (machine learning)")
        
        return creds
    
    except Exception as e:
        logger.error(f"Error reading credentials: {e}")
        return None

def check_bigquery_connection():
    """Check if we can connect to BigQuery"""
    try:
        from google.cloud import bigquery
        
        client = bigquery.Client()
        print(f"\n📊 BIGQUERY CONNECTION")
        print("=" * 30)
        print("✅ BigQuery client created successfully")
        
        # List datasets
        datasets = list(client.list_datasets())
        if datasets:
            print(f"Found {len(datasets)} datasets:")
            for dataset in datasets:
                print(f"  - {dataset.dataset_id}")
        else:
            print("No datasets found")
        
        return client
    
    except ImportError:
        print("❌ google-cloud-bigquery not installed")
        print("Install with: pip install google-cloud-bigquery")
        return None
    except Exception as e:
        print(f"❌ BigQuery connection failed: {e}")
        return None

def check_storage_connection():
    """Check if we can connect to Cloud Storage"""
    try:
        from google.cloud import storage
        
        client = storage.Client()
        print(f"\n💾 CLOUD STORAGE CONNECTION")
        print("=" * 35)
        print("✅ Cloud Storage client created successfully")
        
        # List buckets
        buckets = list(client.list_buckets())
        if buckets:
            print(f"Found {len(buckets)} buckets:")
            for bucket in buckets:
                print(f"  - {bucket.name}")
        else:
            print("No buckets found")
        
        return client
    
    except ImportError:
        print("❌ google-cloud-storage not installed")
        print("Install with: pip install google-cloud-storage")
        return None
    except Exception as e:
        print(f"❌ Cloud Storage connection failed: {e}")
        return None

def check_firestore_connection():
    """Check if we can connect to Firestore"""
    try:
        from google.cloud import firestore
        
        client = firestore.Client()
        print(f"\n🔥 FIRESTORE CONNECTION")
        print("=" * 27)
        print("✅ Firestore client created successfully")
        
        # Try to list collections
        collections = client.collections()
        collection_names = [col.id for col in collections]
        
        if collection_names:
            print(f"Found {len(collection_names)} collections:")
            for name in collection_names[:10]:  # Show first 10
                print(f"  - {name}")
        else:
            print("No collections found")
        
        return client
    
    except ImportError:
        print("❌ google-cloud-firestore not installed")
        print("Install with: pip install google-cloud-firestore")
        return None
    except Exception as e:
        print(f"❌ Firestore connection failed: {e}")
        return None

def main():
    """Main function to inspect GCP setup and connections"""
    print("🏗️  GOOGLE CLOUD PLATFORM CONNECTOR")
    print("=" * 45)
    
    # Load and inspect credentials
    credentials_path = load_gcp_credentials()
    if not credentials_path:
        return
    
    creds = inspect_gcp_credentials(credentials_path)
    if not creds:
        return
    
    # Set the environment variable for GCP client libraries
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    
    # Test connections to various GCP services
    print(f"\n🔌 TESTING GCP SERVICE CONNECTIONS")
    print("=" * 40)
    
    bigquery_client = check_bigquery_connection()
    storage_client = check_storage_connection()
    firestore_client = check_firestore_connection()
    
    # Summary
    print(f"\n📋 CONNECTION SUMMARY")
    print("=" * 25)
    print(f"BigQuery: {'✅ Connected' if bigquery_client else '❌ Failed'}")
    print(f"Cloud Storage: {'✅ Connected' if storage_client else '❌ Failed'}")
    print(f"Firestore: {'✅ Connected' if firestore_client else '❌ Failed'}")
    
    # Recommendations
    print(f"\n💡 NEXT STEPS")
    print("=" * 15)
    if bigquery_client:
        print("1. Use BigQuery to query your training data")
        print("2. Export results for LLM analysis")
    if storage_client:
        print("1. Check Cloud Storage buckets for data files")
        print("2. Download training datasets")
    if firestore_client:
        print("1. Query Firestore collections for invoice data")
        print("2. Export documents for analysis")
    
    print("\n🎯 To see your training data, we can create queries for any connected service!")

if __name__ == "__main__":
    main()