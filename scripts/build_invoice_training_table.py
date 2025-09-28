"""Build an invoice-aligned training table from existing BigQuery tables.

Usage (example):
  poetry run python scripts/build_invoice_training_table.py \
    --project my-gcp-project \
    --source-datasets fraud,raw_data,transactions \
    --dest-dataset training \
    --dest-table invoice_training \
    --dry-run

Notes:
- The script uses Application Default Credentials or the path set in
  the environment variable GOOGLE_APPLICATION_CREDENTIALS.
- It scans tables in the provided source datasets and picks tables whose
  names contain invoice/receipt/transaction/purchase/billing (configurable).
- For each candidate table the script inspects the schema and composes a
  safe SELECT that maps fields to a canonical invoice schema. The result
  is appended to the destination table in BigQuery.
"""
from __future__ import annotations

import argparse
import logging
from typing import List, Optional

from google.cloud import bigquery
from google.api_core.exceptions import NotFound

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def candidate_table(table_id: str, include_keywords: List[str]) -> bool:
    name = table_id.lower()
    return any(k in name for k in include_keywords)


def field_exists(schema, field_name: str) -> bool:
    for f in schema:
        if f.name == field_name:
            return True
    return False


def get_field(schema, field_name: str):
    for f in schema:
        if f.name == field_name:
            return f
    return None


def build_select_sql(project: str, dataset: str, table: str, schema, source_rank: int = 0) -> str:
    """Build a SELECT that maps available fields to canonical names."""
    # Build invoice_id expression from available columns
    id_parts = []
    if field_exists(schema, "invoice_id"):
        id_parts.append("CAST(invoice_id AS STRING)")
    if field_exists(schema, "invoice_number"):
        id_parts.append("CAST(invoice_number AS STRING)")
    if field_exists(schema, "id"):
        id_parts.append("CAST(id AS STRING)")
    if id_parts:
        invoice_id_expr = f"COALESCE({', '.join(id_parts)}, GENERATE_UUID()) AS invoice_id"
    else:
        invoice_id_expr = "GENERATE_UUID() AS invoice_id"

    # vendor: prefer nested vendor.name if present, then vendor_name or vendor
    vendor_field = get_field(schema, "vendor")
    if vendor_field and vendor_field.field_type == 'RECORD':
        # Extract nested name when vendor is a RECORD, else fallback
        vendor_expr = "COALESCE((SELECT v.name FROM UNNEST([vendor]) AS v LIMIT 1), vendor_name, TO_JSON_STRING(vendor), '') AS vendor_name"
    elif field_exists(schema, "vendor_name"):
        vendor_expr = "COALESCE(vendor_name, TO_JSON_STRING(vendor), '') AS vendor_name"
    elif field_exists(schema, "vendor"):
        # vendor exists but not a RECORD (e.g., STRING) - convert safely
        vendor_expr = "COALESCE(CAST(vendor AS STRING), '') AS vendor_name"
    else:
        vendor_expr = "'' AS vendor_name"

    # numeric fields (only reference if present, else NULL)
    subtotal_expr = "SAFE_CAST(subtotal AS FLOAT64) AS subtotal" if field_exists(schema, "subtotal") else "NULL AS subtotal"
    tax_expr = "SAFE_CAST(tax_amount AS FLOAT64) AS tax_amount" if field_exists(schema, "tax_amount") else "NULL AS tax_amount"
    total_expr = "SAFE_CAST(total_amount AS FLOAT64) AS total_amount" if field_exists(schema, "total_amount") else "NULL AS total_amount"

    # date - try several common candidates
    date_candidates = ["invoice_date", "date", "processed_ts", "received_date"]
    date_field = None
    for cand in date_candidates:
        if field_exists(schema, cand):
            date_field = cand
            break
    date_expr = f"SAFE_CAST({date_field} AS STRING) AS invoice_date" if date_field else "NULL AS invoice_date"

    # line items: only if present in schema
    if field_exists(schema, "line_items"):
        line_items_expr = "TO_JSON_STRING(line_items) AS line_items_json"
    else:
        line_items_expr = "NULL AS line_items_json"

    raw_expr = "TO_JSON_STRING(t) AS raw_record"

    select_cols = [
        invoice_id_expr,
        vendor_expr,
        date_expr,
        subtotal_expr,
        tax_expr,
        total_expr,
        line_items_expr,
        raw_expr,
        f"'{project}.{dataset}.{table}' AS source_table",
        f"{source_rank} AS source_rank",
    ]

    cols_sql = ",\n  ".join(select_cols)
    sql = f"SELECT\n  {cols_sql}\nFROM `{project}.{dataset}.{table}` t"
    return sql


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="GCP project id")
    parser.add_argument("--source-datasets", required=True, help="Comma-separated list of datasets to scan")
    parser.add_argument("--dest-dataset", required=True, help="Destination dataset for the training table")
    parser.add_argument("--dest-table", required=True, help="Destination table name for the training table")
    parser.add_argument("--include-keywords", default="invoice,receipt,transaction,purchase,billing", help="Comma-separated keywords to detect candidate tables")
    parser.add_argument("--dry-run", action="store_true", help="Print planned queries but don't execute")
    args = parser.parse_args(argv)

    client = bigquery.Client(project=args.project)

    include_keywords = [k.strip().lower() for k in args.include_keywords.split(",") if k.strip()]
    source_datasets = [d.strip() for d in args.source_datasets.split(",") if d.strip()]

    dest_table_ref = f"{args.project}.{args.dest_dataset}.{args.dest_table}"

    logger.info("Destination table: %s", dest_table_ref)

    # Ensure destination dataset exists
    try:
        client.get_dataset(f"{args.project}.{args.dest_dataset}")
    except NotFound:
        raise RuntimeError(f"Destination dataset {args.dest_dataset} not found in project {args.project}")

    # Create destination table if it doesn't exist (empty with flexible schema)
    try:
        client.get_table(dest_table_ref)
        logger.info("Destination table exists and will be appended to: %s", dest_table_ref)
    except NotFound:
        logger.info("Creating destination table: %s", dest_table_ref)
        schema = [
            bigquery.SchemaField("invoice_id", "STRING"),
            bigquery.SchemaField("vendor_name", "STRING"),
            bigquery.SchemaField("invoice_date", "STRING"),
            bigquery.SchemaField("subtotal", "FLOAT"),
            bigquery.SchemaField("tax_amount", "FLOAT"),
            bigquery.SchemaField("total_amount", "FLOAT"),
            bigquery.SchemaField("line_items_json", "STRING"),
            bigquery.SchemaField("raw_record", "STRING"),
            bigquery.SchemaField("source_table", "STRING"),
        ]
        table = bigquery.Table(dest_table_ref, schema=schema)
        client.create_table(table)
        logger.info("Created table %s", dest_table_ref)

    # For each source dataset, list tables and discover candidates
    appended_any = False
    staging_tables = []
    for ds in source_datasets:
        dataset_ref = client.dataset(ds)
        try:
            tables = list(client.list_tables(dataset_ref))
        except Exception as e:
            logger.warning("Could not list tables in dataset %s: %s", ds, e)
            continue

        for tbl in tables:
            table_id = tbl.table_id
            fq_table = f"{args.project}.{ds}.{table_id}"
            if not candidate_table(table_id, include_keywords):
                continue

            logger.info("Found candidate table: %s", fq_table)

            # Inspect schema
            try:
                table_obj = client.get_table(fq_table)
            except Exception as e:
                logger.warning("Failed to get table %s: %s", fq_table, e)
                continue

            # compute source rank from dataset order
            source_rank = source_datasets.index(ds)
            sql = build_select_sql(args.project, ds, table_id, table_obj.schema, source_rank=source_rank)

            if args.dry_run:
                logger.info("Dry run - generated SQL for %s:\n%s", fq_table, sql)
                continue

            # Append results to destination table
            job_config = bigquery.QueryJobConfig()
            job_config.destination = dest_table_ref
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND

            # Write to a per-source staging table to avoid schema mismatch when appending
            staging_table = f"{args.project}.{args.dest_dataset}.invoice_training_staging__{ds}__{table_id}"
            logger.info("Writing normalized rows for %s into staging table %s", fq_table, staging_table)
            job_config.destination = staging_table
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
            query_job = client.query(sql, job_config=job_config)
            query_job.result()
            logger.info("Wrote staging rows from %s (job: %s)", fq_table, query_job.job_id)
            staging_tables.append(staging_table)
            appended_any = True

    logger.info("ETL staging complete. Staging tables: %s", staging_tables)

    # Consolidation: union all staging tables plus existing destination (if exists), then dedupe by invoice_id
    if appended_any:
        logger.info("Consolidating staging tables into destination with dedupe")

        union_parts = []
        for st in staging_tables:
            # Explicitly cast every column to canonical types so UNION ALL is compatible
            part = (
                "SELECT\n  CAST(invoice_id AS STRING) AS invoice_id,\n  CAST(vendor_name AS STRING) AS vendor_name,\n  CAST(invoice_date AS STRING) AS invoice_date,\n  SAFE_CAST(subtotal AS FLOAT64) AS subtotal,\n  SAFE_CAST(tax_amount AS FLOAT64) AS tax_amount,\n  SAFE_CAST(total_amount AS FLOAT64) AS total_amount,\n  CAST(line_items_json AS STRING) AS line_items_json,\n  CAST(raw_record AS STRING) AS raw_record,\n  CAST(source_table AS STRING) AS source_table,\n  SAFE_CAST(source_rank AS INT64) AS source_rank\nFROM `" + st + "`"
            )
            union_parts.append(part)

        # Include existing destination (if exists) with low priority source_rank
        try:
            client.get_table(dest_table_ref)
            union_parts.append(
                "SELECT\n  CAST(invoice_id AS STRING) AS invoice_id,\n  CAST(vendor_name AS STRING) AS vendor_name,\n  CAST(invoice_date AS STRING) AS invoice_date,\n  SAFE_CAST(subtotal AS FLOAT64) AS subtotal,\n  SAFE_CAST(tax_amount AS FLOAT64) AS tax_amount,\n  SAFE_CAST(total_amount AS FLOAT64) AS total_amount,\n  CAST(line_items_json AS STRING) AS line_items_json,\n  CAST(raw_record AS STRING) AS raw_record,\n  CAST(source_table AS STRING) AS source_table,\n  999 AS source_rank\nFROM `" + dest_table_ref + "`"
            )
        except Exception:
            # destination may not exist yet
            pass

        union_sql = "\nUNION ALL\n".join(union_parts)

        final_sql = f"CREATE OR REPLACE TABLE `{dest_table_ref}` AS\nSELECT invoice_id, vendor_name, invoice_date, subtotal, tax_amount, total_amount, line_items_json, raw_record, source_table FROM (\n  SELECT *, ROW_NUMBER() OVER (PARTITION BY invoice_id ORDER BY source_rank) AS rn FROM (\n    {union_sql}\n  )\n) WHERE rn = 1"

        logger.info("Running consolidation SQL (this will replace destination):")
        # run the consolidation
        cq = client.query(final_sql)
        cq.result()
        logger.info("Consolidation complete - destination replaced: %s", dest_table_ref)

        # Quick verification: report total rows and a small sample
        try:
            count_q = client.query(f"SELECT COUNT(*) AS cnt FROM `{dest_table_ref}`")
            cnt = list(count_q.result())[0].cnt
            logger.info("Destination row count: %s", cnt)

            sample_q = client.query(f"SELECT invoice_id, vendor_name, invoice_date, total_amount, source_table FROM `{dest_table_ref}` LIMIT 10")
            rows = list(sample_q.result())
            logger.info("Sample rows from destination:")
            for r in rows:
                # convert Row to dict for nicer logging
                try:
                    row_dict = dict(r.items())
                except Exception:
                    # fallback for Row without items()
                    row_dict = {k: getattr(r, k) for k in r.keys()}
                logger.info("%s", row_dict)
        except Exception as e:
            logger.warning("Failed to fetch verification results: %s", e)

        # Cleanup: drop staging tables
        for st in staging_tables:
            try:
                client.delete_table(st)
                logger.info("Dropped staging table: %s", st)
            except Exception as e:
                logger.warning("Failed to drop staging table %s: %s", st, e)


if __name__ == "__main__":
    main()
