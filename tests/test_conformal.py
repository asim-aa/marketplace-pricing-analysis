"""Tests for split-conformal prediction intervals."""

import pandas as pd
import pytest

from src.uncertainty.conformal import (
    build_prediction_intervals,
    calculate_absolute_residuals,
    conformal_quantile,
    empirical_coverage,
    split_proper_training_and_conformal,
)


def make_training_sample() -> pd.DataFrame:
    """Create ordered example rows for conformal splitting."""
    return pd.DataFrame(
        {
            "id": list(range(10)),
            "posting_date": pd.date_range(
                "2021-04-01",
                periods=10,
                freq="D",
                tz="UTC",
            ),
            "price": [
                5_000,
                6_000,
                7_000,
                8_000,
                9_000,
                10_000,
                11_000,
                12_000,
                13_000,
                14_000,
            ],
        }
    )


def test_conformal_split_is_chronological() -> None:
    training = make_training_sample()

    proper, conformal = split_proper_training_and_conformal(
        training,
        conformal_fraction=0.20,
    )

    assert len(proper) == 8
    assert len(conformal) == 2
    assert proper["posting_date"].max() < conformal["posting_date"].min()


def test_conformal_split_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        split_proper_training_and_conformal(
            make_training_sample(),
            conformal_fraction=1.0,
        )


def test_absolute_residuals_are_calculated() -> None:
    actual = pd.Series([10_000, 20_000, 30_000])
    predicted = pd.Series([9_000, 23_000, 28_000])

    residuals = calculate_absolute_residuals(
        actual,
        predicted,
    )

    assert residuals.tolist() == [1_000, 3_000, 2_000]


def test_conformal_quantile_uses_higher_residual() -> None:
    residuals = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])

    quantile = conformal_quantile(
        residuals,
        confidence_level=0.80,
    )

    assert quantile == 500.0


def test_prediction_intervals_are_created() -> None:
    predictions = pd.Series([10_000.0, 20_000.0])

    intervals = build_prediction_intervals(
        predictions,
        quantile=2_000.0,
    )

    assert intervals.lower_bound.tolist() == [
        8_000.0,
        18_000.0,
    ]
    assert intervals.upper_bound.tolist() == [
        12_000.0,
        22_000.0,
    ]


def test_lower_bound_respects_price_floor() -> None:
    predictions = pd.Series([1_000.0])

    intervals = build_prediction_intervals(
        predictions,
        quantile=2_000.0,
        minimum_price=500.0,
    )

    assert intervals.lower_bound.iloc[0] == 500.0


def test_empirical_coverage_is_calculated() -> None:
    actual = pd.Series([10.0, 20.0, 30.0, 40.0])
    lower = pd.Series([5.0, 18.0, 35.0, 30.0])
    upper = pd.Series([15.0, 22.0, 45.0, 50.0])

    coverage = empirical_coverage(
        actual,
        lower,
        upper,
    )

    assert coverage == 0.75
