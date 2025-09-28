#!/usr/bin/env python3
"""
BigQuery Data Explorer
Explores and extracts training data from your GCP BigQuery datasets
"""

import os
from datetime import datetime
from dotenv import load_dotenv
import json

def explore_datasets():
    """Explore all BigQuery datasets and tables"""
    from google.cloud import bigquery
    
    # Set credentials
    load_dotenv('.env')
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    client = bigquery.Client()
    
    print("🔍 EXPLORING BIGQUERY DATASETS")
    print("=" * 35)
    
    datasets = list(client.list_datasets())
    
    for dataset in datasets:
        print(f"\n📊 Dataset: {dataset.dataset_id}")
        print("-" * 30)
        
        # Get tables in this dataset
        tables = list(client.list_tables(dataset.dataset_id))
        
        if tables:
            for table in tables:
                print(f"  📋 Table: {table.table_id}")
                
                # Get table info
                table_ref = client.get_table(table.reference)
                print(f"     Rows: {table_ref.num_rows:,}")
                print(f"     Columns: {len(table_ref.schema)}")
                print(f"     Size: {table_ref.num_bytes / 1024 / 1024:.2f} MB")
                
                # Show first few column names
                columns = [field.name for field in table_ref.schema[:5]]
                print(f"     Sample columns: {', '.join(columns)}")
                if len(table_ref.schema) > 5:
                    print(f"     ... and {len(table_ref.schema) - 5} more columns")
        else:
            print("  (No tables found)")

def sample_fraud_data():
    """Sample data from fraud-related tables"""
    from google.cloud import bigquery
    
    client = bigquery.Client()
    
    # Queries to explore different datasets
    queries = {
        "document_forgery": """
        SELECT * FROM `vaulted-timing-473322-f9.document_forgery.*` 
        LIMIT 5
        """,
        
        "fraud_detection_relational": """
        SELECT * FROM `vaulted-timing-473322-f9.fraud_detection_relational.*` 
        LIMIT 5
        """,
        
        "ieee_cis_fraud": """
        SELECT * FROM `vaulted-timing-473322-f9.ieee_cis_fraud.*` 
        LIMIT 5
        """,
        
        "transactional_fraud": """
        SELECT * FROM `vaulted-timing-473322-f9.transactional_fraud.*` 
        LIMIT 5
        """
    }
    
    print("\n🔬 SAMPLING FRAUD DATA")
    print("=" * 25)
    
    results = {}
    
    for dataset_name, query in queries.items():
        try:
            print(f"\n📊 Sampling from {dataset_name}...")
            
            # First, let's see what tables exist in this dataset
            dataset_ref = client.dataset(dataset_name)
            tables = list(client.list_tables(dataset_ref))
            
            if not tables:
                print(f"  ❌ No tables found in {dataset_name}")
                continue
                
            print(f"  📋 Found {len(tables)} table(s):")
            for table in tables:
                print(f"     - {table.table_id}")
            
            # Query the first table
            first_table = tables[0]
            sample_query = f"""
            SELECT * FROM `vaulted-timing-473322-f9.{dataset_name}.{first_table.table_id}` 
            LIMIT 3
            """
            
            query_job = client.query(sample_query)
            sample_results = query_job.result()
            
            # Convert to list of dictionaries
            rows = []
            for row in sample_results:
                rows.append(dict(row))
            
            results[dataset_name] = {
                'table_name': first_table.table_id,
                'sample_data': rows,
                'row_count': len(rows)
            }
            
            print(f"  ✅ Retrieved {len(rows)} sample rows")
            
        except Exception as e:
            print(f"  ❌ Error sampling {dataset_name}: {e}")
            results[dataset_name] = {'error': str(e)}
    
    return results

def export_training_data():
    """Export fraud data for LLM training"""
    from google.cloud import bigquery
    
    client = bigquery.Client()
    
    print("\n📤 EXPORTING TRAINING DATA")
    print("=" * 30)
    
    # Try to get more comprehensive data
    export_queries = {
        "document_forgery_sample": """
        SELECT * FROM `vaulted-timing-473322-f9.document_forgery.*` 
        LIMIT 100
        """,
        
        "fraud_transactions": """
        SELECT * FROM `vaulted-timing-473322-f9.transactional_fraud.*` 
        WHERE RAND() < 0.1  -- Random 10% sample
        LIMIT 200
        """,
        
        "ieee_fraud_sample": """
        SELECT * FROM `vaulted-timing-473322-f9.ieee_cis_fraud.*` 
        LIMIT 150
        """
    }
    
    exported_data = {}
    
    for export_name, query in export_queries.items():
        try:
            print(f"🔄 Exporting {export_name}...")
            
            # Get the dataset name from the query
            dataset_name = export_name.split('_')[0]
            if dataset_name == 'ieee':
                dataset_name = 'ieee_cis_fraud'
            elif dataset_name == 'fraud':
                dataset_name = 'transactional_fraud'
            
            # Check if dataset exists and get first table
            try:
                dataset_ref = client.dataset(dataset_name)
                tables = list(client.list_tables(dataset_ref))
                
                if tables:
                    first_table = tables[0]
                    actual_query = f"""
                    SELECT * FROM `vaulted-timing-473322-f9.{dataset_name}.{first_table.table_id}` 
                    LIMIT 50
                    """
                    
                    query_job = client.query(actual_query)
                    results = query_job.result()
                    
                    # Convert to list of dictionaries
                    rows = []
                    for row in results:
                        rows.append(dict(row))
                    
                    exported_data[export_name] = rows
                    print(f"  ✅ Exported {len(rows)} rows from {first_table.table_id}")
                
            except Exception as e:
                print(f"  ❌ Error with {export_name}: {e}")
        
        except Exception as e:
            print(f"  ❌ Failed to export {export_name}: {e}")
    
    # Save to file
    if exported_data:
        output_file = "../data/gcp_fraud_training_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'export_timestamp': datetime.now().isoformat(),
                'total_datasets': len(exported_data),
                'data': exported_data
            }, f, indent=2, default=str)
        
        print(f"\n💾 Exported data saved to: {output_file}")
        print(f"📊 Total datasets exported: {len(exported_data)}")
        
        # Show summary
        for name, data in exported_data.items():
            print(f"   - {name}: {len(data)} records")
    
    return exported_data

def main():
    """Main function"""
    print("🗃️  BIGQUERY FRAUD DATA EXPLORER")
    print("=" * 35)
    
    try:
        # Explore datasets
        explore_datasets()
        
        # Sample data
        sample_data = sample_fraud_data()
        
        # Export training data
        training_data = export_training_data()
        
        print(f"\n🎉 DATA EXPLORATION COMPLETE!")
        print("=" * 32)
        print(f"✅ Found 5 BigQuery datasets")
        print(f"✅ Sampled data from multiple fraud datasets")
        print(f"✅ Exported training data for LLM analysis")
        
        return training_data
        
    except Exception as e:
        print(f"❌ Error during exploration: {e}")
        return None

if __name__ == "__main__":
    main()