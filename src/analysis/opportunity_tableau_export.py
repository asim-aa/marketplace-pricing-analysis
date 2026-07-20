"""Create the Tableau-ready marketplace opportunity dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ingestion import PROJECT_ROOT
from src.uncertainty.analyze_uncertainty import (
    OPPORTUNITY_EXPORT_PATH,
)

TABLEAU_OPPORTUNITY_PATH = (
    PROJECT_ROOT / "data" / "tableau" / "marketplace_opportunity_dashboard.csv"
)


def assign_confidence_tier(
    interval_width_ratio: pd.Series,
) -> pd.Series:
    """Classify relative interval width into confidence tiers."""
    return pd.cut(
        interval_width_ratio,
        bins=[
            float("-inf"),
            0.50,
            1.00,
            2.00,
            float("inf"),
        ],
        labels=[
            "high confidence",
            "moderate confidence",
            "low confidence",
            "very low confidence",
        ],
        right=False,
        ordered=True,
    )


def assign_opportunity_strength(
    opportunity: pd.Series,
    relative_gap: pd.Series,
) -> pd.Series:
    """Classify the strength of model-relative pricing gaps."""
    strength = pd.Series(
        "market consistent",
        index=opportunity.index,
        dtype="string",
    )

    outside_interval = opportunity.ne("market_consistent")

    strength.loc[outside_interval & relative_gap.lt(0.25)] = "weak signal"

    strength.loc[outside_interval & relative_gap.ge(0.25) & relative_gap.lt(0.50)] = (
        "moderate signal"
    )

    strength.loc[outside_interval & relative_gap.ge(0.50)] = "strong signal"

    return strength


def validate_opportunity_data(
    listings: pd.DataFrame,
) -> None:
    """Validate required fields and unique listing identifiers."""
    required_columns = {
        "id",
        "price",
        "prediction",
        "lower_90",
        "upper_90",
        "width_90",
        "absolute_error",
        "opportunity_90",
        "opportunity_gap_90",
        "relative_opportunity_gap_90",
        "interval_width_ratio_90",
        "manufacturer",
        "model",
        "year",
        "vehicle_age",
        "odometer",
        "type",
        "state",
        "price_band",
        "vehicle_age_band",
        "mileage_band",
        "posting_date",
    }

    missing_columns = sorted(required_columns.difference(listings.columns))

    if missing_columns:
        raise ValueError(
            "Opportunity data is missing required columns: "
            + ", ".join(missing_columns)
        )

    if listings["id"].duplicated().any():
        raise ValueError("Opportunity data contains duplicate listing IDs.")


def build_tableau_opportunity_export(
    listings: pd.DataFrame,
) -> pd.DataFrame:
    """Create business-friendly fields for Tableau."""
    validate_opportunity_data(listings)

    export = listings.copy()

    export["posting_date"] = pd.to_datetime(
        export["posting_date"],
        utc=True,
    )

    export["posting_day"] = export["posting_date"].dt.tz_convert(None).dt.normalize()

    export["price_difference"] = export["prediction"] - export["price"]

    export["price_difference_percent"] = export["price_difference"] / export[
        "prediction"
    ].clip(lower=500.0)

    export["confidence_tier"] = assign_confidence_tier(
        export["interval_width_ratio_90"]
    )

    export["opportunity_strength"] = assign_opportunity_strength(
        opportunity=export["opportunity_90"],
        relative_gap=export["relative_opportunity_gap_90"],
    )

    export["is_potential_opportunity"] = (
        export["opportunity_90"] == "potentially_underpriced"
    )

    export["is_potential_overpricing"] = (
        export["opportunity_90"] == "potentially_overpriced"
    )

    export["is_market_consistent"] = export["opportunity_90"] == "market_consistent"

    export["asking_price_display"] = "$" + export["price"].round().map("{:,.0f}".format)

    export["predicted_price_display"] = "$" + export["prediction"].round().map(
        "{:,.0f}".format
    )

    export["interval_90_display"] = (
        "$"
        + export["lower_90"].round().map("{:,.0f}".format)
        + " – $"
        + export["upper_90"].round().map("{:,.0f}".format)
    )

    export["manufacturer_model_display"] = (
        export["manufacturer"].astype("string").str.title()
        + " "
        + export["model"].astype("string").str.title()
    )

    export["opportunity_rank"] = (
        export["opportunity_gap_90"]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype("int64")
    )

    numeric_columns = [
        "price",
        "prediction",
        "lower_90",
        "upper_90",
        "width_90",
        "absolute_error",
        "opportunity_gap_90",
        "relative_opportunity_gap_90",
        "interval_width_ratio_90",
        "price_difference",
        "price_difference_percent",
    ]

    export[numeric_columns] = export[numeric_columns].replace([np.inf, -np.inf], np.nan)

    selected_columns = [
        "id",
        "posting_date",
        "posting_day",
        "manufacturer",
        "model",
        "manufacturer_model_display",
        "year",
        "vehicle_age",
        "odometer",
        "type",
        "state",
        "price_band",
        "vehicle_age_band",
        "mileage_band",
        "price",
        "prediction",
        "lower_90",
        "upper_90",
        "width_90",
        "absolute_error",
        "price_difference",
        "price_difference_percent",
        "opportunity_90",
        "opportunity_strength",
        "opportunity_gap_90",
        "relative_opportunity_gap_90",
        "opportunity_rank",
        "confidence_tier",
        "interval_width_ratio_90",
        "covered_90",
        "is_potential_opportunity",
        "is_potential_overpricing",
        "is_market_consistent",
        "asking_price_display",
        "predicted_price_display",
        "interval_90_display",
        "lat",
        "long",
        "region",
        "condition",
        "fuel",
        "transmission",
        "drive",
        "title_status",
    ]

    available_columns = [
        column for column in selected_columns if column in export.columns
    ]

    return export[available_columns].copy()


def save_tableau_opportunity_export(
    export: pd.DataFrame,
    output_path: Path = TABLEAU_OPPORTUNITY_PATH,
) -> None:
    """Save the prepared Tableau dataset."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    export.to_csv(
        output_path,
        index=False,
    )


def main() -> None:
    """Build and save the Tableau opportunity dataset."""
    listings = pd.read_csv(
        OPPORTUNITY_EXPORT_PATH,
        parse_dates=["posting_date"],
    )

    export = build_tableau_opportunity_export(listings)

    save_tableau_opportunity_export(export)

    print("Tableau opportunity export completed")
    print(f"Rows: {len(export):,}")
    print(f"Columns: {export.shape[1]}")
    print(f"Potential opportunities: {export['is_potential_opportunity'].sum():,}")
    print(
        f"Potential overpricing signals: {export['is_potential_overpricing'].sum():,}"
    )
    print(f"Saved to: {TABLEAU_OPPORTUNITY_PATH}")


if __name__ == "__main__":
    main()
