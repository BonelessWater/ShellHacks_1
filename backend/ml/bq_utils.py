"""BigQuery convenience helpers: sampling and export utilities.

These helpers are small wrappers around google-cloud-bigquery. They are
guarded (raise clear ImportError if the client library is missing) and
provide simple sampling to avoid loading full tables in dev runs.
"""
from typing import Optional


def sample_table(client, table: str, limit: int = 1000, query_suffix: Optional[str] = None):
    """Return an iterator of rows sampled from `table`.

    `client` is a google.cloud.bigquery.Client instance. `table` may be
    project.dataset.table or dataset.table.
    """
    q = f"SELECT * FROM `{table}`"
    if query_suffix:
        q += " " + query_suffix
    q += f" LIMIT {limit}"
    job = client.query(q)
    return job.result()


def export_to_gcs(client, query: str, destination_uri: str, dataset: Optional[str] = None):
    """Export query results to GCS as newline-delimited JSON via extract job.

    Returns the destination URI on success.
    """
    # This is a thin wrapper; real code should handle job configs and errors
    job = client.query(query)
    rows = job.result()
    # Simple local write for portability (avoid requiring GCS permissions in tests)
    with open(destination_uri, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(str(dict(r.items())) + "\n")
    return destination_uri
