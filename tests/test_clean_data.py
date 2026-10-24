import pandas as pd

from src.clean_data import TARGET, clean


def test_clean_normalizes_headers_and_drops_duplicates():
    raw = pd.DataFrame({
        "ID": [1, 1],
        "PAY_0": [2, 2],
        "default payment next month": [1, 1],
    })
    result = clean(raw)
    assert len(result) == 1
    assert "pay_1" in result.columns
    assert result[TARGET].dtype == "int8"


def test_clean_drops_missing_target():
    raw = pd.DataFrame({
        "ID": [1, 2],
        "default payment next month": [0, None],
    })
    assert clean(raw)[TARGET].tolist() == [0]
