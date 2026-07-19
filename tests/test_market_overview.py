"""Tests for marketplace overview analysis."""

import pandas as pd

from src.analysis.market_overview import (
    add_analysis_features,
    calculate_market_kpis,
    summarize_age_bands,
    summarize_manufacturers,
    summarize_mileage_bands,
)


def make_analysis_sample() -> pd.DataFrame:
    """Create a small cleaned-listing sample for analysis tests."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "price": [10_000, 20_000, 30_000, 40_000],
            "year": [2020, 2016, 2011, 2001],
            "manufacturer": ["ford", "ford", "toyota", "honda"],
            "model": ["f-150", "escape", "camry", "civic"],
            "odometer": [10_000, 60_000, 110_000, 210_000],
            "state": ["ca", "ca", "tx", "ny"],
            "posting_date": pd.to_datetime(
                [
                    "2021-04-10",
                    "2021-04-11",
                    "2021-04-12",
                    "2021-04-13",
                ],
                utc=True,
            ),
        }
    )


def test_add_analysis_features_calculates_vehicle_age() -> None:
    listings = make_analysis_sample()
    analysis = add_analysis_features(listings)

    assert analysis["vehicle_age"].tolist() == [1, 5, 10, 20]


def test_add_analysis_features_assigns_age_bands() -> None:
    listings = make_analysis_sample()
    analysis = add_analysis_features(listings)

    assert analysis["vehicle_age_band"].astype("string").tolist() == [
        "0-3 years",
        "4-7 years",
        "8-12 years",
        "13-20 years",
    ]


def test_add_analysis_features_assigns_mileage_bands() -> None:
    listings = make_analysis_sample()
    analysis = add_analysis_features(listings)

    assert analysis["mileage_band"].astype("string").tolist() == [
        "0-25k",
        "50k-100k",
        "100k-150k",
        "200k+",
    ]


def test_market_kpis_use_median_values() -> None:
    listings = add_analysis_features(make_analysis_sample())
    kpis = calculate_market_kpis(listings)

    assert kpis["listing_count"] == 4
    assert kpis["median_asking_price"] == 25_000
    assert kpis["median_odometer"] == 85_000
    assert kpis["manufacturer_count"] == 3


def test_manufacturer_summary_groups_listings() -> None:
    listings = add_analysis_features(make_analysis_sample())

    summary = summarize_manufacturers(
        listings,
        minimum_listings=1,
    )

    ford = summary.loc[summary["manufacturer"] == "ford"].iloc[0]

    assert ford["listing_count"] == 2
    assert ford["median_price"] == 15_000


def test_age_summary_preserves_ordered_bands() -> None:
    listings = add_analysis_features(make_analysis_sample())
    summary = summarize_age_bands(listings)

    assert summary["vehicle_age_band"].astype("string").tolist() == [
        "0-3 years",
        "4-7 years",
        "8-12 years",
        "13-20 years",
    ]


def test_mileage_summary_counts_all_rows() -> None:
    listings = add_analysis_features(make_analysis_sample())
    summary = summarize_mileage_bands(listings)

    assert summary["listing_count"].sum() == len(listings)
