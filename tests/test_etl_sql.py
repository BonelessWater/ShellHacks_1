from types import SimpleNamespace
from scripts import build_invoice_training_table as etl


def make_field(name, field_type=None):
    # Minimal stand-in for bigquery.SchemaField used by the script
    # Always include field_type attribute (None when unspecified) to mimic
    # bigquery.SchemaField which exposes .name and .field_type
    return SimpleNamespace(name=name, field_type=field_type)


def test_build_select_sql_happy_path_contains_expected_expressions():
    schema = [
        make_field("invoice_id"),
        make_field("vendor", "RECORD"),
        make_field("subtotal"),
        make_field("tax_amount"),
        make_field("total_amount"),
        make_field("line_items"),
        make_field("invoice_date"),
    ]

    sql = etl.build_select_sql("proj", "ds", "tbl", schema, source_rank=1)

    assert "CAST(invoice_id AS STRING)" in sql
    # vendor RECORD path should attempt to extract nested name
    assert "SELECT v.name" in sql or "TO_JSON_STRING(vendor)" in sql
    assert "SAFE_CAST(subtotal AS FLOAT64)" in sql
    assert "SAFE_CAST(tax_amount AS FLOAT64)" in sql
    assert "SAFE_CAST(total_amount AS FLOAT64)" in sql
    assert "TO_JSON_STRING(line_items)" in sql
    assert "'proj.ds.tbl' AS source_table" in sql
    assert "1 AS source_rank" in sql


def test_build_select_sql_missing_ids_uses_generate_uuid():
    schema = [make_field("vendor")]
    sql = etl.build_select_sql("p", "d", "t", schema)
    assert "GENERATE_UUID() AS invoice_id" in sql
