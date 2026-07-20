"""Tests for leakage-safe model feature engineering."""

import pandas as pd
import pytest

from src.features.build_features import (
    MODEL_FEATURES,
    build_model_features,
    build_target,
)


def make_feature_sample() -> pd.DataFrame:
    """Create a small representative vehicle-listing sample."""
    return pd.DataFrame(
        {
            "id": [1, 2],
            "price": [15_000, 25_000],
            "year": [2018, 2021],
            "manufacturer": ["Ford", "Toyota"],
            "model": ["F-150", "Camry"],
            "condition": ["Good", "unknown"],
            "cylinders": ["6 Cylinders", "4 Cylinders"],
            "fuel": ["Gas", "Gas"],
            "odometer": [60_000, 0],
            "title_status": ["Clean", "Clean"],
            "transmission": ["Automatic", "Automatic"],
            "drive": ["4WD", "FWD"],
            "size": ["Full-Size", "Mid-Size"],
            "type": ["Pickup", "Sedan"],
            "paint_color": ["Black", None],
            "region": ["San Diego", "Los Angeles"],
            "state": ["CA", "CA"],
            "lat": [32.7, 34.0],
            "long": [-117.1, -118.2],
            "posting_date": pd.to_datetime(
                ["2021-04-10", "2021-04-11"],
                utc=True,
            ),
        }
    )


def test_build_model_features_returns_expected_columns() -> None:
    features = build_model_features(make_feature_sample())

    assert features.columns.tolist() == MODEL_FEATURES


def test_vehicle_age_is_calculated_from_posting_year() -> None:
    features = build_model_features(make_feature_sample())

    assert features["vehicle_age"].tolist() == [3, 0]


def test_mileage_per_year_handles_new_vehicle() -> None:
    features = build_model_features(make_feature_sample())

    assert features.loc[0, "mileage_per_year"] == 20_000
    assert features.loc[1, "mileage_per_year"] == 0


def test_manufacturer_model_is_normalized() -> None:
    features = build_model_features(make_feature_sample())

    assert features.loc[0, "manufacturer_model"] == "ford__f-150"
    assert features.loc[1, "manufacturer_model"] == "toyota__camry"


def test_missing_optional_category_becomes_unknown() -> None:
    features = build_model_features(make_feature_sample())

    assert features.loc[1, "paint_color"] == "unknown"


def test_posting_day_of_week_is_created() -> None:
    features = build_model_features(make_feature_sample())

    assert features["posting_day_of_week"].tolist() == [
        "saturday",
        "sunday",
    ]


def test_listing_completeness_penalizes_unknown_values() -> None:
    features = build_model_features(make_feature_sample())

    assert (
        features.loc[0, "listing_completeness"]
        > features.loc[1, "listing_completeness"]
    )


def test_build_target_returns_float_prices() -> None:
    target = build_target(make_feature_sample())

    assert target.dtype == "float64"
    assert target.tolist() == [15_000.0, 25_000.0]


def test_missing_required_feature_column_raises_error() -> None:
    listings = make_feature_sample().drop(columns=["manufacturer"])

    with pytest.raises(ValueError):
        build_model_features(listings)


def test_missing_target_raises_error() -> None:
    listings = make_feature_sample().drop(columns=["price"])

    with pytest.raises(ValueError):
        build_target(listings)
