"""Create a Tableau-ready analytical dataset from cleaned vehicle listings."""

from pathlib import Path

import pandas as pd

from src.analysis.market_overview import (
    add_analysis_features,
    load_clean_listings,
)
from src.data.ingestion import PROJECT_ROOT

TABLEAU_OUTPUT_PATH = PROJECT_ROOT / "data" / "tableau" / "marketplace_overview.csv"

TABLEAU_COLUMNS = [
    "id",
    "posting_date",
    "posting_day",
    "price",
    "year",
    "vehicle_age",
    "vehicle_age_band",
    "manufacturer",
    "model",
    "condition",
    "cylinders",
    "fuel",
    "odometer",
    "mileage_per_year",
    "mileage_band",
    "title_status",
    "transmission",
    "drive",
    "size",
    "type",
    "paint_color",
    "region",
    "state",
    "lat",
    "long",
    "repeated_vin",
]


def build_tableau_dataset(
    listings: pd.DataFrame,
) -> pd.DataFrame:
    """Return a Tableau-ready marketplace analysis dataset."""
    analysis = add_analysis_features(listings)

    missing_columns = [
        column for column in TABLEAU_COLUMNS if column not in analysis.columns
    ]

    if missing_columns:
        missing_display = ", ".join(missing_columns)
        raise ValueError(
            f"Tableau export is missing required columns: {missing_display}"
        )

    tableau_data = analysis[TABLEAU_COLUMNS].copy()

    tableau_data["vehicle_age_band"] = tableau_data["vehicle_age_band"].astype("string")

    tableau_data["mileage_band"] = tableau_data["mileage_band"].astype("string")

    tableau_data["posting_date"] = tableau_data["posting_date"].dt.tz_convert(None)

    tableau_data["posting_day"] = tableau_data["posting_day"].dt.tz_convert(None)

    return tableau_data


def save_tableau_dataset(
    tableau_data: pd.DataFrame,
    output_path: Path = TABLEAU_OUTPUT_PATH,
) -> None:
    """Save the Tableau-ready dataset as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tableau_data.to_csv(output_path, index=False)


def summarize_tableau_export(
    tableau_data: pd.DataFrame,
) -> None:
    """Print summary information for the Tableau export."""
    print("Tableau marketplace dataset created successfully")
    print(f"Rows: {len(tableau_data):,}")
    print(f"Columns: {tableau_data.shape[1]}")
    print(f"Duplicate listing IDs: {tableau_data['id'].duplicated().sum():,}")
    print(f"Missing prices: {tableau_data['price'].isna().sum():,}")
    print(f"Missing posting dates: {tableau_data['posting_date'].isna().sum():,}")
    print(f"Output: {TABLEAU_OUTPUT_PATH}")


def main() -> None:
    """Build and save the Tableau-ready marketplace dataset."""
    listings = load_clean_listings()
    tableau_data = build_tableau_dataset(listings)

    save_tableau_dataset(tableau_data)
    summarize_tableau_export(tableau_data)


if __name__ == "__main__":
    main()
