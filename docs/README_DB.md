Of course. Here is a full rundown of the first dataset we loaded together, formatted so you can share it with your teammates.

-----

## \#\# Dataset 1 of 6: Financial Anomaly Data

This is a clean, straightforward dataset that is ideal for initial experiments with anomaly detection algorithms.

README — BigQuery and Database Notes

This repository includes a data pipeline and ML helpers that interact with Google BigQuery. This file documents how the repository now handles unqualified table names, test safety (no-writes by default), and a lightweight TensorFlow shim used for fast unit testing.

Summary of behavior

- Dataset qualification
  - If a fully-qualified identifier is provided (project.dataset.table), it is used as-is.
  - If a `dataset.table` string is provided, the pipeline infers the `project` from the configured credentials and uses `project.dataset.table`.
  - If only `table` is provided, the pipeline qualifies it as `project.default_dataset.table`. The default dataset is resolved from `BQ_DEFAULT_DATASET` or `TEST_BQ_DATASET`, and falls back to `transactional_fraud` when neither is set.

- Tests do not write to BigQuery by default
  - An autouse pytest fixture `prevent_bq_writes` (in `tests/conftest.py`) monkeypatches common write paths so test runs are read-only by default. To allow tests or CI jobs to perform real BigQuery writes, set `ALLOW_REAL_BQ_WRITES=true` in the environment (use with care).

- Lightweight TensorFlow shim
  - To avoid requiring the TensorFlow wheel (large binary) for developers and CI, the repository contains a small pure-Python `tensorflow.py` shim providing the subset of the TF API needed by unit tests: basic `tf.data.Dataset` operations, `tf.io.read_file`, small `tf.image` helpers, `tf.constant`, and a `cast` helper. This keeps test setup fast while preserving semantics required by the unit tests.

- ML preprocessing details
  - The ML dataset builders (in `data_pipeline/integrations/ml_frameworks.py`) now:
    - Drop datetime-like columns before numeric conversion.
    - Convert boolean columns to integer (0/1).
    - One-hot encode categorical features with `pd.get_dummies(...).fillna(0)` and then convert to NumPy float arrays (float32) for model consumption.

Why these changes

Many tests failed at collection/runtime due to heavy imports (TensorFlow), unqualified BigQuery table names that depended on local gcloud config, and dtype conversion errors when creating NumPy arrays from DataFrames containing non-numeric columns. The changes above make the test experience frictionless and safer for developers while preserving real BigQuery reads for integration-style checks.

How to run smoke checks and tests safely

1) Set credentials for read access (use your service-account JSON):

   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

2) Optionally set the default dataset to resolve unqualified table names used in tests:

   export TEST_BQ_DATASET=transactional_fraud
   export BQ_DEFAULT_DATASET=${TEST_BQ_DATASET}

3) Run the read-only BigQuery smoke check:

   poetry run python scripts/smoke_bq.py

   This will run a small read-only query using your configured credentials.

4) Run the test suite (tests are read-only by default):

   poetry run pytest -q

   To enable writes (dangerous), set `ALLOW_REAL_BQ_WRITES=true` in the environment.

Developer tips

- Align gcloud CLI and the Python client project:

  gcloud config set project <project-id-from-service-account-json>

- Inspect datasets available to the configured credentials:

  poetry run python - <<'PY'
  from google.cloud import bigquery
  client = bigquery.Client()
  print('project:', client.project)
  print('datasets:', [d.dataset_id for d in client.list_datasets()])
  PY

- If you want to run with real TensorFlow instead of the shim, install TensorFlow into the Poetry environment and remove or rename the local `tensorflow.py` shim.

CI notes

- The tests now run without requiring TensorFlow and without writing to BigQuery. For integration/e2e CI jobs that should write to BigQuery, set `ALLOW_REAL_BQ_WRITES=true` in a controlled environment and ensure the service account has the right permissions.

- Consider adding an environment mapping in `pipeline_config.yaml` if you have separate datasets per environment.

Next steps

- If you'd like, I can also update the main `README.md` with a short section describing these changes and the recommended developer workflow. Reply if you want me to proceed.

  * **`Amount`** (`FLOAT`): The monetary value of the transaction.
  * **`Class`** (`INTEGER`): The target label. This is the column you want to predict. It is **`1`** for fraudulent transactions and **`0`** for all others.

### \#\#\# How to Access

Your teammates can use the following SQL query to see a sample of 10 confirmed fraudulent transactions.

```sql
-- This query retrieves 10 fraudulent transactions along with their time and amount.
SELECT
  Time,
  Amount,
  V1,
  V2,
  V3
FROM
  `vaulted-timing-473322-f9.transactional_fraud.credit_card_fraud`
WHERE
  Class = 1
LIMIT 10;
```

### \#\#\# Original Source

  * Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

-----

Let me know when you're ready to move on to the next dataset.

Of course. Here is the rundown for the next dataset.

-----

## \#\# Dataset 3 of 6: Fraud Detection Dataset (Relational)

This dataset is more advanced because it mirrors a real-world data warehouse. It's not a single file but a collection of interconnected tables. This structure is excellent for practicing **feature engineering**, which requires you to combine data from multiple sources to create meaningful insights.

### \#\#\# BigQuery Location

The tables are located in their own dedicated dataset. Your teammates can find them under:

```
vaulted-timing-473322-f9.fraud_detection_relational
```

The primary tables you loaded are:

  * `transaction_records`
  * `customer_profiles_customer_data`
  * `merchant_information_merchant_data`
  * And others...

### \#\#\# Data Description

This is a synthetic, relational dataset designed to simulate a real financial system. It contains multiple tables that are linked by common IDs (like `CustomerID` and `MerchantID`). The main table, `transaction_records`, holds the core transaction data, while other tables provide supplementary information about the customers and merchants involved. Its main purpose is to force the development of advanced data joining and feature engineering skills, which are essential in a production environment.

### \#\#\# Key Columns (in `transaction_records`)

  * **`TransactionID`** (`STRING`): A unique identifier for each transaction.
  * **`CustomerID`** (`STRING`): An identifier that links to the `customer_profiles_customer_data` table.
  * **`MerchantID`** (`STRING`): An identifier that links to the `merchant_information_merchant_data` table.
  * **`TransactionAmount`** (`FLOAT`): The value of the transaction.
  * **`Timestamp`** (`TIMESTAMP`): The date and time of the transaction.
  * **`FraudIndicator`** (`BOOLEAN`): The target label, indicating if the transaction was fraudulent (`true`) or not (`false`).

### \#\#\# How to Access

This dataset's power comes from joining the tables. The following query shows your teammates how to combine the transaction data with customer and merchant information to get a complete picture.

```sql
-- This query joins three tables to link transactions with customer and merchant details.
SELECT
  t.TransactionID,
  t.TransactionAmount,
  c.FullName AS CustomerName,
  c.EmailAddress AS CustomerEmail,
  m.MerchantName,
  m.Category AS MerchantCategory
FROM
  `vaulted-timing-473322-f9.fraud_detection_relational.transaction_records` AS t
JOIN
  `vaulted-timing-473322-f9.fraud_detection_relational.customer_profiles_customer_data` AS c
ON
  t.CustomerID = c.CustomerID
JOIN
  `vaulted-timing-473322-f9.fraud_detection_relational.merchant_information_merchant_data` AS m
ON
  t.MerchantID = m.MerchantID
WHERE
  t.FraudIndicator = TRUE
LIMIT 10;

```

### \#\#\# Original Source

  * Kaggle: [Fraud Detection Dataset (by goyaladi)](https://www.kaggle.com/datasets/goyaladi/fraud-detection-dataset)

-----

Ready for the next one when you are.

Of course. Let's move on to the next one. This is the largest and most complex dataset you've loaded.

-----

## \#\# Dataset 4 of 6: IEEE-CIS Fraud Detection

This is a famous, large-scale dataset from a Kaggle competition. It's highly realistic and challenging, making it an excellent resource for building and testing sophisticated fraud detection models.

### \#\#\# BigQuery Location

The two main tables for this dataset are located in their own dedicated dataset:

```
vaulted-timing-473322-f9.ieee_cis_fraud
```

The two tables are:

  * **`transaction`**: Contains the primary transaction data.
  * **`identity`**: Contains identity information (like device type, browser) associated with the transactions.

### \#\#\# Data Description

This dataset contains real-world e-commerce transactions. It's split into two tables that are linked by the `TransactionID` column. The data is rich but also complex; it has many anonymized features (`V` columns), categorical features (`M` columns, `card` columns), and missing values, which accurately reflects the challenges of real-world data. The goal is to predict the `isFraud` label.

### \#\#\# Key Columns

  * **`TransactionID`** (`INTEGER`): The unique ID for each transaction. This is the key to join the two tables.
  * **`isFraud`** (`INTEGER`): The target label. **`1`** for fraudulent transactions, **`0`** for legitimate ones.
  * **`TransactionAmt`** (`FLOAT`): The transaction amount in US dollars.
  * **`ProductCD`** (`STRING`): A code for the product purchased.
  * **`card1` - `card6`** (`STRING` / `INTEGER`): Various anonymized features about the payment card, such as card type, category, and issuing bank.
  * **`id_01` - `id_38`** (`FLOAT` / `STRING`): Anonymized identity features found in the `identity` table.
  * **`DeviceInfo`** (`STRING`): Information about the device used for the transaction.

### \#\#\# How to Access

To use this dataset effectively, you must **JOIN** the `transaction` and `identity` tables. This query shows your teammates how to combine them to get a complete view for training a model.

```sql
-- This query joins the transaction and identity tables to create a complete dataset.
SELECT
  t.TransactionID,
  t.isFraud,
  t.TransactionAmt,
  t.ProductCD,
  t.card4 AS CardType,
  i.DeviceInfo
FROM
  `vaulted-timing-473322-f9.ieee_cis_fraud.transaction` AS t
LEFT JOIN
  `vaulted-timing-473322-f9.ieee_cis_fraud.identity` AS i
ON
  t.TransactionID = i.TransactionID
WHERE
  t.isFraud = 1
ORDER BY
  t.TransactionAmt DESC
LIMIT 15;
```

### \#\#\# Original Source

  * Kaggle Competition: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data)

-----

Let me know when you're ready for the image datasets.

Of course. We'll now move on to the datasets designed for image analysis. These are fundamentally different because the primary data (the images themselves) are stored in Google Cloud Storage, while the tables in BigQuery contain the *metadata* about those images.

-----

## \#\# Dataset 5 of 6: Roboflow Invoice Data (Object Detection)

This dataset is designed for the first step in an image forgery pipeline: **training a computer vision model to find and identify key information zones** on an invoice document.

### \#\#\# Primary Data Location (Images)

The invoice image files (`.jpg`) are not in BigQuery. They are stored in your Google Cloud Storage bucket at the following path:

```
gs://vaulted-timing-473322-f9-invoicedata/invoice_images/Invoice_data.v10i.yolov11/
```

### \#\#\# BigQuery Location (Annotations)

The annotations for these images, which describe the location of each labeled bounding box, were processed and loaded into a single table at:

```
vaulted-timing-473322-f9.document_forgery.invoice_annotations
```

### \#\#\# Data Description

This is a public object detection dataset containing thousands of invoice images. Each image has been annotated with bounding boxes that identify seven distinct classes (e.g., `CompanyName`, `InvoiceDetails`, `TableDetails`). Its purpose is to train a model, like YOLO or a Vision Transformer, to automatically locate these regions on any given invoice. This is a critical prerequisite for targeted text extraction (OCR) and subsequent forgery analysis.

### \#\#\# Key Columns

  * **`image_gcs_path`** (`STRING`): The full path to the corresponding image file in Google Cloud Storage.
  * **`class_name`** (`STRING`): The name of the labeled object (e.g., "CustomerDetails").
  * **`class_id`** (`INTEGER`): The numerical ID for the class name.
  * **`x_center_norm`**, **`y_center_norm`** (`FLOAT`): The normalized center coordinates (x, y) of the bounding box.
  * **`width_norm`**, **`height_norm`** (`FLOAT`): The normalized width and height of the bounding box.

### \#\#\# How to Access

Your teammates can query this table to get the labels for a specific image. This query, for example, finds all the labeled regions for one of the training images.

```sql
-- This query retrieves all bounding box annotations for a specific invoice image.
SELECT
  class_name,
  x_center_norm,
  y_center_norm,
  width_norm,
  height_norm
FROM
  `vaulted-timing-473322-f9.document_forgery.invoice_annotations`
WHERE
  image_gcs_path = 'gs://vaulted-timing-473322-f9-invoicedata/invoice_images/Invoice_data.v10i.yolov11/train/images/0a2b53f6-4a45-4252-8178-319b0d1e5763_jpg.rf.0003b71f39160f4e38e3a2c5a01340a6.jpg'
```

### \#\#\# Original Source

  * Roboflow Universe: [Invoice\_data Dataset](https://universe.roboflow.com/personal-17k7d/invoice_data-puetu)

-----

Let me know when you're ready for the final dataset.

Of course. You have now loaded all the data. Here is the final dataset rundown for your team.

-----

## \#\# Dataset 6 of 6: Receipt Dataset for Forgery Detection (L3i-Share)

This is a highly specialized academic dataset designed for training models to perform two tasks: **forgery localization** (finding the exact pixels that were manipulated) and **fraud classification** (labeling a whole document as real or fake).

### \#\#\# Primary Data Location (Images)

The receipt image files (`.jpg`, `.png`) are stored in your Google Cloud Storage bucket at the following path:

```
gs://vaulted-timing-473322-f9-invoicedata/receipt_forgery_dataset/findit/
```

### \#\#\# BigQuery Location (Annotations)

The annotations for this dataset were processed from XML files and loaded into three separate tables within the `document_forgery` dataset:

1.  **`receipt_forgery_annotations`**: Contains the bounding box coordinates of manipulated regions for the **training** images.
2.  **`receipt_forgery_annotations_test`**: Contains the bounding box coordinates for the **test** images.
3.  **`receipt_fraud_classification`**: Contains a simple `true`/`false` label for every image in both the train and test sets, indicating if it's fraudulent.

### \#\#\# Data Description

This dataset was created by the Université de La Rochelle for a fraud detection competition. It contains a mix of genuine receipt images and receipts that have been realistically altered. Crucially, it provides ground truth annotations in two forms:

  * **For Task 1 (Classification):** A master list of which filenames correspond to fraudulent receipts.
  * **For Task 2 (Localization):** For the fraudulent receipts, it provides the exact pixel coordinates (x, y, width, height) of every manipulated area.

This makes it a premier resource for training and evaluating supervised models that can both classify a document and pinpoint the exact location of a forgery.

### \#\#\# Key Columns

  * In **`receipt_forgery_annotations`**:
      * `image_gcs_path` (`STRING`): The full path to the image file in GCS.
      * `is_fraudulent` (`BOOLEAN`): A label indicating if the image contains a forgery.
      * `forgery_x`, `forgery_y`, `forgery_width`, `forgery_height` (`INTEGER`): The pixel coordinates of a forged bounding box. (These are `NULL` for genuine receipts).
  * In **`receipt_fraud_classification`**:
      * `image_filename` (`STRING`): The name of the image file.
      * `is_fraudulent` (`BOOLEAN`): The label indicating if the image is a forgery.
      * `dataset_split` (`STRING`): Indicates if the image belongs to the 'train' or 'test' set.

### \#\#\# How to Access

Your teammates can use this query to find all the fraudulent receipts in the training set and join them with their forgery locations.

```sql
-- This query finds all fraudulent receipts in the training set and lists the locations of their forgeries.
SELECT
  class.image_filename,
  loc.forgery_x,
  loc.forgery_y,
  loc.forgery_width,
  loc.forgery_height
FROM
  `vaulted-timing-473322-f9.document_forgery.receipt_fraud_classification` AS class
JOIN
  `vaulted-timing-473322-f9.document_forgery.receipt_forgery_annotations` AS loc
ON
  class.image_filename = loc.image_filename
WHERE
  class.is_fraudulent = TRUE
  AND class.dataset_split = 'train'
LIMIT 15;
```

### \#\#\# Original Source

  * L3i-Share Portal, Université de La Rochelle: [http://l3i-share.univ-lr.fr/](http://l3i-share.univ-lr.fr/)

-----

You have now successfully loaded and documented all the datasets from your research. Your BigQuery project is fully set up for the next stage of your fraud detection project.