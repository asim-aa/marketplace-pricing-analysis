"""Tests for vehicle pricing baseline models."""

import pandas as pd
import pytest

from src.models.baselines import (
    GlobalMedianBaseline,
    SegmentMedianBaseline,
)


def make_baseline_sample() -> pd.DataFrame:
    """Create a small training sample for baseline tests."""
    rows = []

    for index in range(5):
        rows.append(
            {
                "id": index,
                "price": 10_000 + index * 1_000,
                "year": 2018,
                "manufacturer": "ford",
                "model": "f-150",
                "odometer": 50_000 + index * 1_000,
                "posting_date": pd.Timestamp(
                    "2021-04-10",
                    tz="UTC",
                ),
            }
        )

    for index in range(20):
        rows.append(
            {
                "id": 100 + index,
                "price": 20_000 + index * 100,
                "year": 2015,
                "manufacturer": "toyota",
                "model": f"model-{index}",
                "odometer": 80_000 + index * 500,
                "posting_date": pd.Timestamp(
                    "2021-04-10",
                    tz="UTC",
                ),
            }
        )

    return pd.DataFrame(rows)


def test_global_median_baseline_predicts_one_value() -> None:
    training = make_baseline_sample()

    model = GlobalMedianBaseline().fit(training)
    predictions = model.predict(training.head(3))

    assert predictions.nunique() == 1
    assert predictions.iloc[0] == training["price"].median()


def test_global_median_requires_fit() -> None:
    model = GlobalMedianBaseline()

    with pytest.raises(ValueError):
        model.predict(make_baseline_sample())


def test_segment_baseline_uses_detailed_segment() -> None:
    training = make_baseline_sample()

    model = SegmentMedianBaseline(
        minimum_detailed_count=5,
        minimum_manufacturer_age_count=20,
    ).fit(training)

    test = pd.DataFrame(
        {
            "id": [999],
            "price": [0],
            "year": [2018],
            "manufacturer": ["ford"],
            "model": ["f-150"],
            "odometer": [55_000],
            "posting_date": pd.to_datetime(
                ["2021-04-11"],
                utc=True,
            ),
        }
    )

    prediction = model.predict(test).iloc[0]

    assert prediction == 12_000


def test_segment_baseline_falls_back_from_sparse_detail() -> None:
    training = make_baseline_sample()

    model = SegmentMedianBaseline(
        minimum_detailed_count=5,
        minimum_manufacturer_age_count=20,
    ).fit(training)

    test = pd.DataFrame(
        {
            "id": [999],
            "price": [0],
            "year": [2015],
            "manufacturer": ["toyota"],
            "model": ["unseen-model"],
            "odometer": [90_000],
            "posting_date": pd.to_datetime(
                ["2021-04-11"],
                utc=True,
            ),
        }
    )

    prediction = model.predict(test).iloc[0]

    expected = training.loc[
        training["manufacturer"] == "toyota",
        "price",
    ].median()

    assert prediction == expected


def test_segment_baseline_uses_global_fallback() -> None:
    training = make_baseline_sample()

    model = SegmentMedianBaseline().fit(training)

    test = pd.DataFrame(
        {
            "id": [999],
            "price": [0],
            "year": [2015],
            "manufacturer": ["unknown-brand"],
            "model": ["unknown-model"],
            "odometer": [75_000],
            "posting_date": pd.to_datetime(
                ["2021-04-11"],
                utc=True,
            ),
        }
    )

    prediction = model.predict(test).iloc[0]

    assert prediction == training["price"].median()


def test_segment_baseline_requires_fit() -> None:
    model = SegmentMedianBaseline()

    with pytest.raises(ValueError):
        model.predict(make_baseline_sample())
