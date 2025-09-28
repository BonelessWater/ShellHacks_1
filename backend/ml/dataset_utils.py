"""Dataset utilities: fingerprinting and simple checks.

Fingerprinting helps ensure train/test separation and reproducibility.
"""
import hashlib
from typing import Iterable


def fingerprint_rows(rows: Iterable[Iterable]) -> str:
    """Compute a stable fingerprint for a sequence of rows.

    Each row should be an iterable of primitive values (numbers/strings).
    Returns a hex digest string.
    """
    h = hashlib.sha256()
    for row in rows:
        # Create a stable representation
        row_str = ",".join(str(x) for x in row)
        h.update(row_str.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def fingerprint_csv(path: str, delimiter: str = ",") -> str:
    """Fingerprint a CSV by rows. Lightweight (streams file).
    """
    import csv

    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        return fingerprint_rows(reader)
