"""Tests for the Tableau opportunity export."""

import pandas as pd
import pytest

from src.analysis.opportunity_tableau_export import (
    assign_confidence_tier,
    assign_opportunity_strength,
    build_tableau_opportunity_export,
)


def make_opportunity_sample() -> pd.DataFrame:
    """Create representative opportunity rows."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "posting_date": pd.to_datetime(
                [
                    "2021-05-01",
                    "2021-05-02",
                    "2021-05-03",
                ],
                utc=True,
            ),
            "price": [
                8_000.0,
                15_000.0,
                30_000.0,
            ],
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
            "opportunity_90": [
                "potentially_underpriced",
                "market_consistent",
                "potentially_overpriced",
            ],
            "opportunity_gap_90": [
                2_000.0,
                0.0,
                5_000.0,
            ],
            "relative_opportunity_gap_90": [
                0.20,
                0.00,
                0.50,
            ],
            "interval_width_ratio_90": [
                0.40,
                0.80,
                2.20,
            ],
            "covered_90": [
                False,
                True,
                False,
            ],
            "manufacturer": [
                "ford",
                "toyota",
                "honda",
            ],
            "model": [
                "focus",
                "camry",
                "pilot",
            ],
            "year": [
                2018.0,
                2015.0,
                2020.0,
            ],
            "vehicle_age": [
                3.0,
                6.0,
                1.0,
            ],
            "odometer": [
                40_000.0,
                80_000.0,
                15_000.0,
            ],
            "type": [
                "sedan",
                "sedan",
                "suv",
            ],
            "state": ["ca", "ca", "tx"],
            "price_band": [
                "$5k-$10k",
                "$10k-$20k",
                "$20k-$35k",
            ],
            "vehicle_age_band": [
                "3-5 years",
                "6-10 years",
                "0-2 years",
            ],
            "mileage_band": [
                "25k-50k",
                "50k-100k",
                "Under 25k",
            ],
        }
    )


def test_confidence_tiers_are_assigned() -> None:
    values = pd.Series([0.25, 0.75, 1.50, 2.50])

    tiers = assign_confidence_tier(values)

    assert tiers.astype("string").tolist() == [
        "high confidence",
        "moderate confidence",
        "low confidence",
        "very low confidence",
    ]


def test_opportunity_strength_is_assigned() -> None:
    opportunity = pd.Series(
        [
            "potentially_underpriced",
            "potentially_overpriced",
            "market_consistent",
        ]
    )

    relative_gap = pd.Series([0.20, 0.60, 0.00])

    strength = assign_opportunity_strength(
        opportunity,
        relative_gap,
    )

    assert strength.tolist() == [
        "weak signal",
        "strong signal",
        "market consistent",
    ]


def test_tableau_export_creates_business_fields() -> None:
    export = build_tableau_opportunity_export(make_opportunity_sample())

    assert "confidence_tier" in export.columns
    assert "opportunity_strength" in export.columns
    assert "price_difference" in export.columns
    assert "interval_90_display" in export.columns


def test_underpriced_flag_is_created() -> None:
    export = build_tableau_opportunity_export(make_opportunity_sample())

    assert export["is_potential_opportunity"].tolist() == [True, False, False]


def test_duplicate_ids_are_rejected() -> None:
    sample = make_opportunity_sample()
    sample.loc[1, "id"] = sample.loc[0, "id"]

    with pytest.raises(ValueError):
        build_tableau_opportunity_export(sample)
