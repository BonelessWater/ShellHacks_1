import os
from backend.ml import dataset_utils


def test_fingerprint_rows_basic():
    rows = [[1, 2, "a"], [3, 4, "b"]]
    fp1 = dataset_utils.fingerprint_rows(rows)
    # Recomputing with same data should be identical
    rows2 = [[1, 2, "a"], [3, 4, "b"]]
    fp2 = dataset_utils.fingerprint_rows(rows2)
    assert fp1 == fp2


def test_fingerprint_csv(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,b,c\n1,2,x\n3,4,y\n")
    fp = dataset_utils.fingerprint_csv(str(p))
    assert isinstance(fp, str) and len(fp) > 0
