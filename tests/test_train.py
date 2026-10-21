from pathlib import Path
import pandas as pd

def test_dataset_has_expected_columns():
    path = Path(__file__).resolve().parents[1] / "Data" / "drug.csv"
    if not path.exists():
        # Dataset is intentionally downloaded separately by the learner.
        return
    df = pd.read_csv(path)
    assert "Drug" in df.columns
    assert len(df) >= 10
