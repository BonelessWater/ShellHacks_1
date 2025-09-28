#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Create BigQuery datasets
bq mk --dataset --location=US vaulted-timing-473322-f9:feature_store
bq mk --dataset --location=US vaulted-timing-473322-f9:data_monitoring
bq mk --dataset --location=US vaulted-timing-473322-f9:versioned_data
bq mk --dataset --location=US vaulted-timing-473322-f9:streaming_results

# Set up authentication
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"

# Initialize feature store
python -c "from data_pipeline import easy_access; easy_access.quick_start()"

# Run tests
pytest tests/

echo "✅ Data pipeline setup complete!"