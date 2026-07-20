"""Tests for conditional coverage and opportunity analysis."""

import pandas as pd
import pytest

from src.uncertainty.analyze_uncertainty import (
    add_segment_columns,
    classify_opportunities,
    summarize_conditional_coverage,
)


def make_scored_sample() -> pd.DataFrame:
    """Create representative scored marketplace listings."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "price": [8_000.0, 15_000.0, 30_000.0],
            "prediction": [
                15_000.0,
                15_000.0,
                20_000.0,
            ],
            "lower_90": [
                10_000.0,
                10_000.0,
                15_000.0,
            ],
            "upper_90": [
                20_000.0,
                20_000.0,
                25_000.0,
            ],
            "width_90": [
                10_000.0,
                10_000.0,
                10_000.0,
            ],
            "absolute_error": [
                7_000.0,
                0.0,
                10_000.0,
            ],
            "year": [2018.0, 2010.0, 2000.0],
            "odometer": [
                20_000.0,
                80_000.0,
                220_000.0,
            ],
            "posting_date": pd.to_datetime(
                [
                    "2021-05-01",
                    "2021-05-01",
                    "2021-05-01",
                ],
                utc=True,
            ),
            "manufacturer": [
                "ford",
                "toyota",
                "honda",
            ],
            "type": ["sedan", "suv", "pickup"],
            "state": ["ca", "ca", "tx"],
        }
    )


def test_opportunity_classification() -> None:
    result = classify_opportunities(make_scored_sample())

    assert result["opportunity_90"].tolist() == [
        "potentially_underpriced",
        "market_consistent",
        "potentially_overpriced",
    ]


def test_opportunity_gap_is_positive_outside_interval() -> None:
    result = classify_opportunities(make_scored_sample())

    assert result["opportunity_gap_90"].tolist() == [
        2_000.0,
        0.0,
        5_000.0,
    ]


def test_coverage_indicator_is_created() -> None:
    result = classify_opportunities(make_scored_sample())

    assert result["covered_90"].tolist() == [
        False,
        True,
        False,
    ]


def test_segment_columns_are_created() -> None:
    result = add_segment_columns(make_scored_sample())

    assert "price_band" in result.columns
    assert "vehicle_age_band" in result.columns
    assert "mileage_band" in result.columns


def test_conditional_coverage_summary() -> None:
    scored = classify_opportunities(make_scored_sample())

    summary = summarize_conditional_coverage(
        scored=scored,
        segment_column="state",
        minimum_rows=1,
    )

    california = summary[summary["segment_value"] == "ca"].iloc[0]

    assert california["listing_count"] == 2
    assert california["empirical_coverage"] == 0.5


def test_conditional_coverage_rejects_invalid_minimum() -> None:
    scored = classify_opportunities(make_scored_sample())

    with pytest.raises(ValueError):
        summarize_conditional_coverage(
            scored=scored,
            segment_column="state",
            minimum_rows=0,
        )
