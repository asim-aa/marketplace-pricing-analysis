"""Tests for the Tableau marketplace export."""

import pandas as pd

from src.analysis.tableau_export import (
    TABLEAU_COLUMNS,
    build_tableau_dataset,
)


def make_tableau_sample() -> pd.DataFrame:
    """Return a small cleaned dataset for Tableau export tests."""
    return pd.DataFrame(
        {
            "id": [1, 2],
            "price": [15_000, 25_000],
            "year": [2018, 2015],
            "manufacturer": ["ford", "toyota"],
            "model": ["f-150", "camry"],
            "condition": ["good", "excellent"],
            "cylinders": ["6 cylinders", "4 cylinders"],
            "fuel": ["gas", "gas"],
            "odometer": [60_000, 90_000],
            "title_status": ["clean", "clean"],
            "transmission": ["automatic", "automatic"],
            "VIN": ["VIN1", "VIN2"],
            "drive": ["4wd", "fwd"],
            "size": ["full-size", "mid-size"],
            "type": ["pickup", "sedan"],
            "paint_color": ["black", "white"],
            "region": ["san diego", "los angeles"],
            "state": ["ca", "ca"],
            "lat": [32.7, 34.0],
            "long": [-117.1, -118.2],
            "posting_date": pd.to_datetime(
                ["2021-04-10", "2021-04-11"],
                utc=True,
            ),
            "repeated_vin": [False, False],
        }
    )


def test_tableau_export_contains_expected_columns() -> None:
    tableau_data = build_tableau_dataset(make_tableau_sample())

    assert tableau_data.columns.tolist() == TABLEAU_COLUMNS


def test_tableau_export_preserves_row_count() -> None:
    listings = make_tableau_sample()
    tableau_data = build_tableau_dataset(listings)

    assert len(tableau_data) == len(listings)


def test_tableau_export_adds_analysis_features() -> None:
    tableau_data = build_tableau_dataset(make_tableau_sample())

    assert "vehicle_age" in tableau_data.columns
    assert "vehicle_age_band" in tableau_data.columns
    assert "mileage_per_year" in tableau_data.columns
    assert "mileage_band" in tableau_data.columns


def test_tableau_export_removes_timezone_information() -> None:
    tableau_data = build_tableau_dataset(make_tableau_sample())

    assert tableau_data["posting_date"].dt.tz is None
    assert tableau_data["posting_day"].dt.tz is None


def test_tableau_export_has_no_missing_core_fields() -> None:
    tableau_data = build_tableau_dataset(make_tableau_sample())

    assert (
        tableau_data[
            [
                "id",
                "price",
                "year",
                "manufacturer",
                "model",
                "posting_date",
            ]
        ]
        .isna()
        .sum()
        .sum()
        == 0
    )
