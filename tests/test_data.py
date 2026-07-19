"""Tests for the vehicle data ingestion, validation, and cleaning pipeline."""

import pandas as pd

from src.data.cleaning import clean_listings
from src.data.validation import add_validation_flags


def make_sample_listings() -> pd.DataFrame:
    """Return a small dataset containing valid and invalid listings."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "price": [15_000, 0, 20_000],
            "year": [2018, 2015, None],
            "manufacturer": ["Ford", "Toyota", "Honda"],
            "model": ["F-150", "Camry", "Civic"],
            "condition": ["Excellent", None, "Good"],
            "cylinders": [None, "4 Cylinders", "4 Cylinders"],
            "fuel": ["Gas", "Gas", "Gas"],
            "odometer": [50_000, 80_000, 70_000],
            "title_status": ["Clean", "Clean", "Clean"],
            "transmission": ["Automatic", "Automatic", "Automatic"],
            "VIN": ["VIN1", "VIN2", "VIN3"],
            "drive": ["4WD", None, "FWD"],
            "size": [None, "Mid-Size", "Compact"],
            "type": ["Pickup", "Sedan", "Sedan"],
            "paint_color": ["Black", None, "Blue"],
            "region": ["San Diego", "Los Angeles", "Sacramento"],
            "state": ["CA", "CA", "CA"],
            "lat": [32.7, 34.0, 38.5],
            "long": [-117.1, -118.2, -121.5],
            "posting_date": pd.to_datetime(
                [
                    "2021-04-10",
                    "2021-04-11",
                    "2021-04-12",
                ],
                utc=True,
            ),
        }
    )


def test_validation_flags_invalid_price() -> None:
    listings = make_sample_listings()
    validated = add_validation_flags(listings)
    assert not validated.loc[0, "invalid_price"]
    assert validated.loc[1, "invalid_price"]


def test_validation_flags_missing_year() -> None:
    listings = make_sample_listings()
    validated = add_validation_flags(listings)

    assert validated.loc[2, "invalid_year"]


def test_core_validation_combines_failures() -> None:
    listings = make_sample_listings()
    validated = add_validation_flags(listings)

    assert validated["fails_core_validation"].tolist() == [
        False,
        True,
        True,
    ]


def test_cleaning_removes_invalid_rows() -> None:
    listings = make_sample_listings()
    cleaned = clean_listings(listings)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["id"] == 1


def test_cleaning_normalizes_text() -> None:
    listings = make_sample_listings()
    cleaned = clean_listings(listings)

    assert cleaned.iloc[0]["manufacturer"] == "ford"
    assert cleaned.iloc[0]["model"] == "f-150"
    assert cleaned.iloc[0]["condition"] == "excellent"


def test_cleaning_fills_optional_categories() -> None:
    listings = make_sample_listings()
    cleaned = clean_listings(listings)

    assert cleaned.iloc[0]["cylinders"] == "unknown"
    assert cleaned.iloc[0]["size"] == "unknown"


def test_cleaned_data_is_sorted_by_posting_date() -> None:
    listings = make_sample_listings().iloc[[2, 0, 1]].copy()
    cleaned = clean_listings(listings)

    assert cleaned["posting_date"].is_monotonic_increasing
