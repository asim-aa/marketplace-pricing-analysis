"""Load and perform structural checks on the raw vehicle listings dataset."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "vehicles.csv"

REQUIRED_COLUMNS = {
    "id",
    "price",
    "year",
    "manufacturer",
    "model",
    "condition",
    "cylinders",
    "fuel",
    "odometer",
    "title_status",
    "transmission",
    "VIN",
    "drive",
    "size",
    "type",
    "paint_color",
    "region",
    "state",
    "lat",
    "long",
    "posting_date",
}

SELECTED_COLUMNS = [
    "id",
    "price",
    "year",
    "manufacturer",
    "model",
    "condition",
    "cylinders",
    "fuel",
    "odometer",
    "title_status",
    "transmission",
    "VIN",
    "drive",
    "size",
    "type",
    "paint_color",
    "region",
    "state",
    "lat",
    "long",
    "posting_date",
]


def validate_raw_file(path: Path = RAW_DATA_PATH) -> None:
    """Confirm that the raw dataset exists before attempting to load it."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Download vehicles.csv and place it in data/raw/."
        )


def inspect_columns(path: Path = RAW_DATA_PATH) -> list[str]:
    """Read only the header and return the available column names."""
    validate_raw_file(path)

    header = pd.read_csv(path, nrows=0)
    return header.columns.tolist()


def validate_required_columns(available_columns: list[str]) -> None:
    """Raise an error when required fields are missing."""
    missing_columns = REQUIRED_COLUMNS.difference(available_columns)

    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing_display}")


def load_raw_listings(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load project-relevant columns from the raw Craigslist dataset."""
    available_columns = inspect_columns(path)
    validate_required_columns(available_columns)

    listings = pd.read_csv(
        path,
        usecols=SELECTED_COLUMNS,
        low_memory=False,
    )

    listings["posting_date"] = pd.to_datetime(
        listings["posting_date"],
        errors="coerce",
        utc=True,
    )

    return listings


def summarize_listings(listings: pd.DataFrame) -> None:
    """Print a small structural summary without cleaning the data."""
    print("Raw vehicle listings loaded successfully")
    print(f"Rows: {len(listings):,}")
    print(f"Columns: {listings.shape[1]}")
    print(f"Duplicate listing IDs: {listings['id'].duplicated().sum():,}")
    print(f"Missing posting dates: {listings['posting_date'].isna().sum():,}")
    print(f"Earliest posting date: {listings['posting_date'].min()}")
    print(f"Latest posting date: {listings['posting_date'].max()}")


def main() -> None:
    """Load the raw dataset and print its structural summary."""
    listings = load_raw_listings()
    summarize_listings(listings)


if __name__ == "__main__":
    main()
