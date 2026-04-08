import pandas as pd

from src.drift_detection.schema import ADULT_CATEGORICAL_FEATURES, ADULT_NUMERIC_FEATURES
from src.orchestration.data_context import (
    load_feature_matrix,
    split_labeled_reference_current,
    split_reference_current,
)


def _feature_df(n: int) -> pd.DataFrame:
    cols_num = list(ADULT_NUMERIC_FEATURES)
    cols_cat = list(ADULT_CATEGORICAL_FEATURES)
    rows = []
    for i in range(n):
        rows.append(
            {
                "age": 25 + i,
                "fnlwgt": 100000 + i,
                "education.num": 10,
                "capital.gain": 0,
                "capital.loss": 0,
                "hours.per.week": 40,
                "workclass": "Private",
                "education": "Bachelors",
                "marital.status": "Never-married",
                "occupation": "Tech-support",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "native.country": "United-States",
            }
        )
    df = pd.DataFrame(rows)
    return df[cols_num + cols_cat]


def test_split_reference_current_incoming_csv_uses_external_batch(tmp_path) -> None:
    incoming = _feature_df(5)
    incoming_path = tmp_path / "incoming.csv"
    incoming.to_csv(incoming_path, index=False)

    X = load_feature_matrix()
    ref, cur = split_reference_current(
        X,
        test_size=0.3,
        random_state=42,
        scenario="incoming_csv",
        current_csv_path=str(incoming_path),
    )

    assert len(ref) > 0
    assert len(cur) == 5
    assert set(cur.columns) == set(incoming.columns)


def test_split_labeled_reference_current_incoming_csv_without_target_returns_none(tmp_path) -> None:
    incoming = _feature_df(4)
    incoming_path = tmp_path / "incoming_no_target.csv"
    incoming.to_csv(incoming_path, index=False)

    ref, cur = split_labeled_reference_current(
        test_size=0.3,
        random_state=42,
        scenario="incoming_csv",
        current_csv_path=str(incoming_path),
    )

    assert ref is not None
    assert cur is None
