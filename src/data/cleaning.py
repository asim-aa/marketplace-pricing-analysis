"""Clean validated vehicle listings and save the analytical dataset."""

from pathlib import Path

import pandas as pd

from src.data.ingestion import PROJECT_ROOT, load_raw_listings
from src.data.validation import add_validation_flags

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_listings.parquet"

CATEGORICAL_COLUMNS = [
    "manufacturer",
    "model",
    "condition",
    "cylinders",
    "fuel",
    "title_status",
    "transmission",
    "drive",
    "size",
    "type",
    "paint_color",
    "region",
    "state",
]

OPTIONAL_CATEGORICAL_COLUMNS = [
    "condition",
    "cylinders",
    "fuel",
    "title_status",
    "transmission",
    "drive",
    "size",
    "type",
    "paint_color",
]


def normalize_text_columns(listings: pd.DataFrame) -> pd.DataFrame:
    """Standardize categorical text while preserving missing values."""
    cleaned = listings.copy()

    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].astype("string").str.strip().str.lower()

    return cleaned


def fill_optional_categories(listings: pd.DataFrame) -> pd.DataFrame:
    """Replace missing optional categories with an explicit unknown label."""
    cleaned = listings.copy()

    for column in OPTIONAL_CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].fillna("unknown")

    return cleaned


def clean_listings(listings: pd.DataFrame) -> pd.DataFrame:
    """Remove core validation failures and standardize retained listings."""
    validated = add_validation_flags(listings)

    cleaned = validated.loc[~validated["fails_core_validation"]].copy()

    cleaned = normalize_text_columns(cleaned)
    cleaned = fill_optional_categories(cleaned)

    validation_columns = [
        "invalid_price",
        "invalid_year",
        "invalid_odometer",
        "missing_manufacturer",
        "missing_model",
        "invalid_posting_date",
        "fails_core_validation",
    ]

    cleaned = cleaned.drop(columns=validation_columns)

    cleaned = cleaned.sort_values(
        by=["posting_date", "id"],
        kind="stable",
    ).reset_index(drop=True)

    return cleaned


def save_clean_listings(
    listings: pd.DataFrame,
    output_path: Path = PROCESSED_DATA_PATH,
) -> None:
    """Save the cleaned analytical dataset as Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    listings.to_parquet(output_path, index=False)


def summarize_cleaning(
    raw_listings: pd.DataFrame,
    cleaned_listings: pd.DataFrame,
) -> None:
    """Print row-retention and cleaned-dataset statistics."""
    removed_rows = len(raw_listings) - len(cleaned_listings)
    retained_percentage = len(cleaned_listings) / len(raw_listings) * 100

    print("Vehicle listing cleaning summary")
    print(f"Raw rows: {len(raw_listings):,}")
    print(f"Removed rows: {removed_rows:,}")
    print(f"Clean rows: {len(cleaned_listings):,}")
    print(f"Rows retained: {retained_percentage:.2f}%")
    print(f"Clean columns: {cleaned_listings.shape[1]}")
    print(f"Output: {PROCESSED_DATA_PATH}")


def main() -> None:
    """Run the complete raw-to-clean dataset pipeline."""
    raw_listings = load_raw_listings()
    cleaned_listings = clean_listings(raw_listings)

    save_clean_listings(cleaned_listings)
    summarize_cleaning(raw_listings, cleaned_listings)


if __name__ == "__main__":
    main()
