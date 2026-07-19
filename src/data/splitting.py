"""Create chronological train, calibration, and test datasets."""

from pathlib import Path

import pandas as pd

from src.analysis.market_overview import load_clean_listings
from src.data.ingestion import PROJECT_ROOT

SPLIT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

TRAIN_PATH = SPLIT_DATA_DIR / "train.parquet"
CALIBRATION_PATH = SPLIT_DATA_DIR / "calibration.parquet"
TEST_PATH = SPLIT_DATA_DIR / "test.parquet"

TRAIN_FRACTION = 0.70
CALIBRATION_FRACTION = 0.15
TEST_FRACTION = 0.15


def validate_split_fractions(
    train_fraction: float,
    calibration_fraction: float,
    test_fraction: float,
) -> None:
    """Confirm that split fractions are positive and sum to one."""
    fractions = [
        train_fraction,
        calibration_fraction,
        test_fraction,
    ]

    if any(fraction <= 0 for fraction in fractions):
        raise ValueError("Every split fraction must be greater than zero.")

    if not abs(sum(fractions) - 1.0) < 1e-9:
        raise ValueError("Split fractions must sum to 1.0.")


def find_time_boundaries(
    listings: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    calibration_fraction: float = CALIBRATION_FRACTION,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Find timestamp boundaries for chronological dataset partitions."""
    ordered = listings.sort_values(
        ["posting_date", "id"],
        kind="stable",
    ).reset_index(drop=True)

    train_end_position = int(len(ordered) * train_fraction)
    calibration_end_position = int(
        len(ordered) * (train_fraction + calibration_fraction)
    )

    calibration_start = ordered.loc[
        train_end_position,
        "posting_date",
    ]

    test_start = ordered.loc[
        calibration_end_position,
        "posting_date",
    ]

    return calibration_start, test_start


def remove_cross_split_vin_overlap(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove later listings whose VIN already appeared in an earlier split."""
    train_vins = set(train["VIN"].dropna())

    calibration = calibration.loc[
        calibration["VIN"].isna() | ~calibration["VIN"].isin(train_vins)
    ].copy()

    earlier_vins = train_vins | set(calibration["VIN"].dropna())

    test = test.loc[test["VIN"].isna() | ~test["VIN"].isin(earlier_vins)].copy()

    return train, calibration, test


def create_chronological_splits(
    listings: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    calibration_fraction: float = CALIBRATION_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create ordered splits and prevent VIN overlap across them."""
    validate_split_fractions(
        train_fraction,
        calibration_fraction,
        test_fraction,
    )

    ordered = listings.sort_values(
        ["posting_date", "id"],
        kind="stable",
    ).reset_index(drop=True)

    calibration_start, test_start = find_time_boundaries(
        ordered,
        train_fraction=train_fraction,
        calibration_fraction=calibration_fraction,
    )

    train = ordered.loc[ordered["posting_date"] < calibration_start].copy()

    calibration = ordered.loc[
        (ordered["posting_date"] >= calibration_start)
        & (ordered["posting_date"] < test_start)
    ].copy()

    test = ordered.loc[ordered["posting_date"] >= test_start].copy()

    train, calibration, test = remove_cross_split_vin_overlap(
        train,
        calibration,
        test,
    )

    train = train.reset_index(drop=True)
    calibration = calibration.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return train, calibration, test


def validate_chronology(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Confirm that no later split precedes an earlier split."""
    if train.empty or calibration.empty or test.empty:
        raise ValueError("Train, calibration, and test splits must be non-empty.")

    if train["posting_date"].max() >= calibration["posting_date"].min():
        raise ValueError("Training and calibration dates overlap.")

    if calibration["posting_date"].max() >= test["posting_date"].min():
        raise ValueError("Calibration and test dates overlap.")


def validate_vin_separation(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Confirm that non-null VINs do not occur across multiple splits."""
    train_vins = set(train["VIN"].dropna())
    calibration_vins = set(calibration["VIN"].dropna())
    test_vins = set(test["VIN"].dropna())

    if train_vins & calibration_vins:
        raise ValueError("VIN overlap exists between train and calibration.")

    if train_vins & test_vins:
        raise ValueError("VIN overlap exists between train and test.")

    if calibration_vins & test_vins:
        raise ValueError("VIN overlap exists between calibration and test.")


def save_splits(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path = SPLIT_DATA_DIR,
) -> None:
    """Save each dataset split as a Parquet file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(output_dir / "train.parquet", index=False)
    calibration.to_parquet(
        output_dir / "calibration.parquet",
        index=False,
    )
    test.to_parquet(output_dir / "test.parquet", index=False)


def summarize_splits(
    original: pd.DataFrame,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Print split sizes, ranges, and VIN leakage information."""
    retained = len(train) + len(calibration) + len(test)
    removed = len(original) - retained

    print("Chronological dataset split created successfully")
    print(f"Original rows: {len(original):,}")
    print(f"Train rows: {len(train):,}")
    print(f"Calibration rows: {len(calibration):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Cross-split VIN rows removed: {removed:,}")
    print()

    for name, split in [
        ("Train", train),
        ("Calibration", calibration),
        ("Test", test),
    ]:
        print(f"{name}: {split['posting_date'].min()} to {split['posting_date'].max()}")


def main() -> None:
    """Create, validate, save, and summarize the data splits."""
    listings = load_clean_listings()

    train, calibration, test = create_chronological_splits(listings)

    validate_chronology(train, calibration, test)
    validate_vin_separation(train, calibration, test)

    save_splits(train, calibration, test)
    summarize_splits(listings, train, calibration, test)


if __name__ == "__main__":
    main()
