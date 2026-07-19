"""Tests for chronological dataset splitting."""

import pandas as pd
import pytest

from src.data.splitting import (
    create_chronological_splits,
    validate_chronology,
    validate_split_fractions,
    validate_vin_separation,
)


def make_split_sample() -> pd.DataFrame:
    """Create ordered listings with one VIN repeated across time."""
    return pd.DataFrame(
        {
            "id": list(range(1, 11)),
            "posting_date": pd.date_range(
                "2021-04-01",
                periods=10,
                freq="D",
                tz="UTC",
            ),
            "VIN": [
                "VIN1",
                "VIN2",
                "VIN3",
                "VIN4",
                "VIN5",
                "VIN6",
                "VIN7",
                "VIN1",
                "VIN8",
                None,
            ],
            "price": [
                10_000,
                11_000,
                12_000,
                13_000,
                14_000,
                15_000,
                16_000,
                17_000,
                18_000,
                19_000,
            ],
        }
    )


def test_split_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        validate_split_fractions(0.70, 0.20, 0.20)


def test_chronological_split_produces_nonempty_sets() -> None:
    listings = make_split_sample()

    train, calibration, test = create_chronological_splits(
        listings,
        train_fraction=0.60,
        calibration_fraction=0.20,
        test_fraction=0.20,
    )

    assert not train.empty
    assert not calibration.empty
    assert not test.empty


def test_split_dates_are_chronological() -> None:
    listings = make_split_sample()

    train, calibration, test = create_chronological_splits(
        listings,
        train_fraction=0.60,
        calibration_fraction=0.20,
        test_fraction=0.20,
    )

    validate_chronology(train, calibration, test)

    assert train["posting_date"].max() < calibration["posting_date"].min()
    assert calibration["posting_date"].max() < test["posting_date"].min()


def test_repeated_vin_is_removed_from_later_split() -> None:
    listings = make_split_sample()

    train, calibration, test = create_chronological_splits(
        listings,
        train_fraction=0.60,
        calibration_fraction=0.20,
        test_fraction=0.20,
    )

    assert "VIN1" in set(train["VIN"].dropna())
    assert "VIN1" not in set(calibration["VIN"].dropna())
    assert "VIN1" not in set(test["VIN"].dropna())


def test_vins_are_separated_across_splits() -> None:
    listings = make_split_sample()

    train, calibration, test = create_chronological_splits(
        listings,
        train_fraction=0.60,
        calibration_fraction=0.20,
        test_fraction=0.20,
    )

    validate_vin_separation(train, calibration, test)


def test_split_preserves_null_vin_rows() -> None:
    listings = make_split_sample()

    train, calibration, test = create_chronological_splits(
        listings,
        train_fraction=0.60,
        calibration_fraction=0.20,
        test_fraction=0.20,
    )

    combined = pd.concat([train, calibration, test])

    assert combined["VIN"].isna().sum() == 1
