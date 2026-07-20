"""Tests for the Ridge vehicle-price model."""

import pandas as pd
import pytest

from src.models.ridge import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    RidgePriceModel,
)


def make_ridge_sample() -> pd.DataFrame:
    """Create a small representative vehicle-listing dataset."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "price": [
                10_000,
                12_000,
                15_000,
                20_000,
                25_000,
                30_000,
            ],
            "year": [2010, 2012, 2014, 2016, 2018, 2020],
            "manufacturer": [
                "ford",
                "ford",
                "toyota",
                "toyota",
                "honda",
                "honda",
            ],
            "model": [
                "focus",
                "fusion",
                "camry",
                "rav4",
                "civic",
                "accord",
            ],
            "condition": ["good"] * 6,
            "cylinders": ["4 cylinders"] * 6,
            "fuel": ["gas"] * 6,
            "odometer": [
                150_000,
                130_000,
                100_000,
                80_000,
                50_000,
                20_000,
            ],
            "title_status": ["clean"] * 6,
            "transmission": ["automatic"] * 6,
            "VIN": [
                "VIN1",
                "VIN2",
                "VIN3",
                "VIN4",
                "VIN5",
                "VIN6",
            ],
            "drive": ["fwd"] * 6,
            "size": ["mid-size"] * 6,
            "type": ["sedan", "sedan", "sedan", "suv", "sedan", "sedan"],
            "paint_color": ["black"] * 6,
            "region": ["san diego"] * 6,
            "state": ["ca"] * 6,
            "lat": [32.7] * 6,
            "long": [-117.1] * 6,
            "posting_date": pd.to_datetime(
                [
                    "2021-04-01",
                    "2021-04-02",
                    "2021-04-03",
                    "2021-04-04",
                    "2021-04-05",
                    "2021-04-06",
                ],
                utc=True,
            ),
        }
    )


def test_ridge_feature_lists_are_complete() -> None:
    assert set(MODEL_FEATURES) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def test_ridge_prepare_features_returns_expected_columns() -> None:
    listings = make_ridge_sample()

    features = RidgePriceModel.prepare_features(listings)

    assert features.columns.tolist() == MODEL_FEATURES
    assert len(features) == len(listings)


def test_ridge_fits_and_predicts() -> None:
    listings = make_ridge_sample()

    model = RidgePriceModel(
        minimum_category_frequency=1,
    ).fit(listings)

    predictions = model.predict(listings)

    assert len(predictions) == len(listings)
    assert predictions.notna().all()


def test_ridge_predictions_respect_price_floor() -> None:
    listings = make_ridge_sample()

    model = RidgePriceModel(
        minimum_category_frequency=1,
    ).fit(listings)

    predictions = model.predict(listings)

    assert (predictions >= 500).all()


def test_ridge_handles_unseen_categories() -> None:
    training = make_ridge_sample()

    model = RidgePriceModel(
        minimum_category_frequency=1,
    ).fit(training)

    unseen = training.head(1).copy()
    unseen["manufacturer"] = "unseen-brand"
    unseen["model"] = "unseen-model"
    unseen["state"] = "tx"

    prediction = model.predict(unseen)

    assert len(prediction) == 1
    assert prediction.notna().all()


def test_ridge_requires_fit_before_prediction() -> None:
    model = RidgePriceModel()

    with pytest.raises(ValueError):
        model.predict(make_ridge_sample())


def test_ridge_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError):
        RidgePriceModel(alpha=0)
