"""Tests for the CatBoost vehicle-price model."""

import pandas as pd
import pytest

from src.models.catboost_model import CatBoostPriceModel


def make_catboost_sample() -> pd.DataFrame:
    """Create a small representative dataset for CatBoost tests."""
    return pd.DataFrame(
        {
            "id": list(range(12)),
            "price": [
                8_000,
                9_000,
                10_000,
                12_000,
                14_000,
                16_000,
                18_000,
                20_000,
                22_000,
                25_000,
                28_000,
                32_000,
            ],
            "year": [
                2010,
                2011,
                2012,
                2013,
                2014,
                2015,
                2016,
                2017,
                2018,
                2019,
                2020,
                2021,
            ],
            "manufacturer": ["ford"] * 6 + ["toyota"] * 6,
            "model": ["focus"] * 3 + ["f-150"] * 3 + ["camry"] * 3 + ["tacoma"] * 3,
            "condition": ["good"] * 12,
            "cylinders": ["4 cylinders"] * 12,
            "fuel": ["gas"] * 12,
            "odometer": [
                160_000,
                150_000,
                140_000,
                130_000,
                120_000,
                110_000,
                90_000,
                80_000,
                70_000,
                50_000,
                30_000,
                10_000,
            ],
            "title_status": ["clean"] * 12,
            "transmission": ["automatic"] * 12,
            "drive": ["fwd"] * 12,
            "size": ["mid-size"] * 12,
            "type": ["sedan"] * 6 + ["pickup"] * 6,
            "paint_color": ["black"] * 12,
            "region": ["san diego"] * 12,
            "state": ["ca"] * 12,
            "lat": [32.7] * 12,
            "long": [-117.1] * 12,
            "posting_date": pd.date_range(
                "2021-04-01",
                periods=12,
                freq="D",
                tz="UTC",
            ),
        }
    )


def test_catboost_fits_and_predicts() -> None:
    listings = make_catboost_sample()

    model = CatBoostPriceModel(
        iterations=20,
        depth=4,
        learning_rate=0.1,
        verbose=False,
    ).fit(
        listings.iloc[:9],
        listings.iloc[9:],
        early_stopping_rounds=5,
    )

    predictions = model.predict(listings.iloc[9:])

    assert len(predictions) == 3
    assert predictions.notna().all()


def test_catboost_predictions_respect_price_floor() -> None:
    listings = make_catboost_sample()

    model = CatBoostPriceModel(
        iterations=10,
        depth=3,
        verbose=False,
    ).fit(listings)

    predictions = model.predict(listings)

    assert (predictions >= 500).all()


def test_catboost_requires_fit_before_prediction() -> None:
    model = CatBoostPriceModel(verbose=False)

    with pytest.raises(ValueError):
        model.predict(make_catboost_sample())


def test_catboost_rejects_invalid_iterations() -> None:
    with pytest.raises(ValueError):
        CatBoostPriceModel(iterations=0)


def test_catboost_feature_importance_contains_all_features() -> None:
    listings = make_catboost_sample()

    model = CatBoostPriceModel(
        iterations=10,
        depth=3,
        verbose=False,
    ).fit(listings)

    importance = model.feature_importance()

    assert len(importance) == 21
    assert set(importance.columns) == {"feature", "importance"}
