"""Build leakage-safe features for vehicle asking-price models."""

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "vehicle_age",
    "odometer",
    "mileage_per_year",
    "lat",
    "long",
    "listing_completeness",
]

CATEGORICAL_FEATURES = [
    "manufacturer",
    "model",
    "manufacturer_model",
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
    "posting_day_of_week",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

COMPLETENESS_COLUMNS = [
    "condition",
    "cylinders",
    "fuel",
    "title_status",
    "transmission",
    "drive",
    "size",
    "type",
    "paint_color",
    "lat",
    "long",
]


def validate_feature_inputs(listings: pd.DataFrame) -> None:
    """Confirm that all raw columns required for feature creation exist."""
    required_columns = {
        "year",
        "odometer",
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
        "lat",
        "long",
        "posting_date",
    }

    missing_columns = sorted(required_columns.difference(listings.columns))

    if missing_columns:
        missing_display = ", ".join(missing_columns)
        raise ValueError(
            f"Feature input is missing required columns: {missing_display}"
        )


def normalize_categorical_features(
    listings: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize categorical values and replace missing values."""
    featured = listings.copy()

    categorical_columns = [
        column for column in CATEGORICAL_FEATURES if column in featured.columns
    ]

    for column in categorical_columns:
        featured[column] = (
            featured[column]
            .astype("string")
            .fillna("unknown")
            .str.strip()
            .str.lower()
            .replace("", "unknown")
        )

    return featured


def calculate_listing_completeness(
    listings: pd.DataFrame,
) -> pd.Series:
    """Count how many optional listing attributes are meaningfully present."""
    completeness = pd.Series(
        0,
        index=listings.index,
        dtype="int64",
    )

    for column in COMPLETENESS_COLUMNS:
        values = listings[column]

        if pd.api.types.is_string_dtype(values):
            present = (
                values.notna()
                & values.astype("string").str.strip().ne("")
                & values.astype("string").str.lower().ne("unknown")
            )
        else:
            present = values.notna()

        completeness = completeness.add(
            present.astype("int64"),
            fill_value=0,
        )

    return completeness.astype("int64")


def build_model_features(
    listings: pd.DataFrame,
) -> pd.DataFrame:
    """Create leakage-safe numeric and categorical model features."""
    validate_feature_inputs(listings)

    featured = listings.copy()

    posting_year = featured["posting_date"].dt.year

    featured["vehicle_age"] = (posting_year - featured["year"]).clip(lower=0)

    safe_vehicle_age = featured["vehicle_age"].replace(0, 1)

    featured["mileage_per_year"] = featured["odometer"] / safe_vehicle_age

    featured["mileage_per_year"] = featured["mileage_per_year"].replace(
        [np.inf, -np.inf], np.nan
    )

    featured["listing_completeness"] = calculate_listing_completeness(featured)

    featured["posting_day_of_week"] = featured["posting_date"].dt.day_name()

    featured["manufacturer_model"] = (
        featured["manufacturer"].astype("string").fillna("unknown")
        + "__"
        + featured["model"].astype("string").fillna("unknown")
    )

    featured = normalize_categorical_features(featured)

    return featured[MODEL_FEATURES].copy()


def build_target(listings: pd.DataFrame) -> pd.Series:
    """Return the asking-price target as floating-point values."""
    if "price" not in listings.columns:
        raise ValueError("Target column `price` is missing.")

    target = listings["price"].astype("float64")

    if target.isna().any():
        raise ValueError("Target column contains missing prices.")

    if (target <= 0).any():
        raise ValueError("Target prices must be greater than zero.")

    return target.rename("price")
