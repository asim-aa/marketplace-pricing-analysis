"""Create data-quality flags for raw vehicle listings."""

import pandas as pd

from src.data.ingestion import load_raw_listings

MIN_VALID_PRICE = 500
MAX_VALID_PRICE = 250_000
MIN_VALID_YEAR = 1900
MAX_VALID_YEAR = 2022
MAX_VALID_ODOMETER = 500_000


def add_validation_flags(listings: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the listings with explicit quality flags."""
    validated = listings.copy()

    validated["invalid_price"] = (
        validated["price"].isna()
        | (validated["price"] < MIN_VALID_PRICE)
        | (validated["price"] > MAX_VALID_PRICE)
    )

    validated["invalid_year"] = (
        validated["year"].isna()
        | (validated["year"] < MIN_VALID_YEAR)
        | (validated["year"] > MAX_VALID_YEAR)
    )

    validated["invalid_odometer"] = (
        validated["odometer"].isna()
        | (validated["odometer"] < 0)
        | (validated["odometer"] > MAX_VALID_ODOMETER)
    )

    validated["missing_manufacturer"] = validated["manufacturer"].isna()
    validated["missing_model"] = validated["model"].isna()
    validated["invalid_posting_date"] = validated["posting_date"].isna()

    validated["repeated_vin"] = validated["VIN"].notna() & validated["VIN"].duplicated(
        keep=False
    )

    required_quality_flags = [
        "invalid_price",
        "invalid_year",
        "invalid_odometer",
        "missing_manufacturer",
        "missing_model",
        "invalid_posting_date",
    ]

    validated["fails_core_validation"] = validated[required_quality_flags].any(axis=1)

    return validated


def summarize_validation(validated: pd.DataFrame) -> None:
    """Print counts for each validation flag."""
    flag_columns = [
        "invalid_price",
        "invalid_year",
        "invalid_odometer",
        "missing_manufacturer",
        "missing_model",
        "invalid_posting_date",
        "repeated_vin",
        "fails_core_validation",
    ]

    print("Vehicle listing validation summary")
    print(f"Total rows: {len(validated):,}")

    for column in flag_columns:
        count = int(validated[column].sum())
        percentage = count / len(validated) * 100

        print(f"{column}: {count:,} ({percentage:.2f}%)")


def main() -> None:
    """Load listings, add validation flags, and print the summary."""
    listings = load_raw_listings()
    validated = add_validation_flags(listings)
    summarize_validation(validated)


if __name__ == "__main__":
    main()
