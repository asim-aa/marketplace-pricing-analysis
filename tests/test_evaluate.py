"""Tests for regression evaluation utilities."""

import pandas as pd

from src.models.evaluate import (
    calculate_regression_metrics,
    evaluate_by_price_band,
)


def test_regression_metrics_are_calculated_correctly() -> None:
    actual = pd.Series([10_000, 20_000, 30_000])
    predicted = pd.Series([9_000, 22_000, 27_000])

    metrics = calculate_regression_metrics(actual, predicted)

    assert metrics["mae"] == 2_000
    assert metrics["median_absolute_error"] == 2_000
    assert round(metrics["rmse"], 2) == 2_160.25


def test_price_band_evaluation_counts_all_rows() -> None:
    actual = pd.Series([4_000, 8_000, 15_000, 30_000, 45_000, 60_000])
    predicted = pd.Series([5_000, 7_000, 14_000, 28_000, 40_000, 55_000])

    summary = evaluate_by_price_band(actual, predicted)

    assert summary["listing_count"].sum() == len(actual)


def test_price_band_evaluation_reports_absolute_error() -> None:
    actual = pd.Series([4_000, 4_500])
    predicted = pd.Series([3_000, 5_000])

    summary = evaluate_by_price_band(actual, predicted)

    under_5k = summary.loc[summary["price_band"] == "Under $5k"].iloc[0]

    assert under_5k["mae"] == 750
    assert under_5k["median_absolute_error"] == 750
