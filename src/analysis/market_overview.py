"""Generate high-level marketplace summaries from cleaned vehicle listings."""

from pathlib import Path

import pandas as pd

from src.data.ingestion import PROJECT_ROOT

CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_listings.parquet"

REPORT_DATA_DIR = PROJECT_ROOT / "reports" / "tables"


def load_clean_listings(
    path: Path = CLEAN_DATA_PATH,
) -> pd.DataFrame:
    """Load the cleaned analytical vehicle-listing dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at {path}. "
            "Run `uv run python -m src.data.cleaning` first."
        )

    return pd.read_parquet(path)


def add_analysis_features(listings: pd.DataFrame) -> pd.DataFrame:
    """Create simple fields used for exploratory market analysis."""
    analysis = listings.copy()

    analysis["vehicle_age"] = (
        analysis["posting_date"].dt.year - analysis["year"]
    ).clip(lower=0)

    analysis["mileage_per_year"] = analysis["odometer"] / analysis[
        "vehicle_age"
    ].replace(0, 1)

    analysis["posting_day"] = analysis["posting_date"].dt.floor("D")

    analysis["vehicle_age_band"] = pd.cut(
        analysis["vehicle_age"],
        bins=[-1, 3, 7, 12, 20, float("inf")],
        labels=[
            "0-3 years",
            "4-7 years",
            "8-12 years",
            "13-20 years",
            "21+ years",
        ],
    )

    analysis["mileage_band"] = pd.cut(
        analysis["odometer"],
        bins=[
            -1,
            25_000,
            50_000,
            100_000,
            150_000,
            200_000,
            float("inf"),
        ],
        labels=[
            "0-25k",
            "25k-50k",
            "50k-100k",
            "100k-150k",
            "150k-200k",
            "200k+",
        ],
    )

    return analysis


def calculate_market_kpis(
    listings: pd.DataFrame,
) -> pd.Series:
    """Calculate top-level marketplace KPIs."""
    return pd.Series(
        {
            "listing_count": len(listings),
            "median_asking_price": listings["price"].median(),
            "mean_asking_price": listings["price"].mean(),
            "median_odometer": listings["odometer"].median(),
            "median_vehicle_year": listings["year"].median(),
            "median_vehicle_age": listings["vehicle_age"].median(),
            "manufacturer_count": listings["manufacturer"].nunique(),
            "model_count": listings["model"].nunique(),
            "state_count": listings["state"].nunique(),
        },
        name="value",
    )


def summarize_manufacturers(
    listings: pd.DataFrame,
    minimum_listings: int = 500,
) -> pd.DataFrame:
    """Summarize listing volume and pricing by manufacturer."""
    summary = (
        listings.groupby("manufacturer", observed=True)
        .agg(
            listing_count=("id", "count"),
            median_price=("price", "median"),
            mean_price=("price", "mean"),
            median_odometer=("odometer", "median"),
            median_year=("year", "median"),
            median_vehicle_age=("vehicle_age", "median"),
        )
        .reset_index()
    )

    summary = summary.loc[summary["listing_count"] >= minimum_listings]

    return summary.sort_values(
        "listing_count",
        ascending=False,
    ).reset_index(drop=True)


def summarize_vehicle_types(
    listings: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize marketplace listings by vehicle type."""
    return (
        listings.groupby("type", observed=True)
        .agg(
            listing_count=("id", "count"),
            median_price=("price", "median"),
            median_odometer=("odometer", "median"),
            median_vehicle_age=("vehicle_age", "median"),
        )
        .reset_index()
        .sort_values("listing_count", ascending=False)
        .reset_index(drop=True)
    )


def summarize_states(
    listings: pd.DataFrame,
    minimum_listings: int = 1_000,
) -> pd.DataFrame:
    """Summarize listing volume and asking prices by state."""
    summary = (
        listings.groupby("state", observed=True)
        .agg(
            listing_count=("id", "count"),
            median_price=("price", "median"),
            median_odometer=("odometer", "median"),
            median_vehicle_age=("vehicle_age", "median"),
        )
        .reset_index()
    )

    summary = summary.loc[summary["listing_count"] >= minimum_listings]

    return summary.sort_values(
        "listing_count",
        ascending=False,
    ).reset_index(drop=True)


def summarize_age_bands(
    listings: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize asking prices by vehicle-age band."""
    return (
        listings.groupby("vehicle_age_band", observed=True)
        .agg(
            listing_count=("id", "count"),
            median_price=("price", "median"),
            median_odometer=("odometer", "median"),
        )
        .reset_index()
    )


def summarize_mileage_bands(
    listings: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize asking prices by mileage band."""
    return (
        listings.groupby("mileage_band", observed=True)
        .agg(
            listing_count=("id", "count"),
            median_price=("price", "median"),
            median_vehicle_age=("vehicle_age", "median"),
        )
        .reset_index()
    )


def summarize_daily_listings(
    listings: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize daily listing volume and asking prices."""
    return (
        listings.groupby("posting_day", observed=True)
        .agg(
            listing_count=("id", "count"),
            median_price=("price", "median"),
        )
        .reset_index()
        .sort_values("posting_day")
        .reset_index(drop=True)
    )


def save_analysis_tables(
    tables: dict[str, pd.DataFrame | pd.Series],
    output_dir: Path = REPORT_DATA_DIR,
) -> None:
    """Save analysis summaries as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        output_path = output_dir / f"{name}.csv"

        if isinstance(table, pd.Series):
            table.to_frame().to_csv(output_path)
        else:
            table.to_csv(output_path, index=False)


def main() -> None:
    """Run the initial marketplace overview analysis."""
    listings = load_clean_listings()
    analysis = add_analysis_features(listings)

    tables = {
        "market_kpis": calculate_market_kpis(analysis),
        "manufacturer_summary": summarize_manufacturers(analysis),
        "vehicle_type_summary": summarize_vehicle_types(analysis),
        "state_summary": summarize_states(analysis),
        "vehicle_age_summary": summarize_age_bands(analysis),
        "mileage_summary": summarize_mileage_bands(analysis),
        "daily_listing_summary": summarize_daily_listings(analysis),
    }

    save_analysis_tables(tables)

    print("Marketplace overview created successfully")
    print()
    print(tables["market_kpis"])
    print()
    print("Top manufacturers by listing count:")
    print(tables["manufacturer_summary"].head(10).to_string(index=False))
    print()
    print(f"Analysis tables saved to: {REPORT_DATA_DIR}")


if __name__ == "__main__":
    main()
