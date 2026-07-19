"""Create exploratory marketplace visualizations."""


import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.market_overview import (
    REPORT_DATA_DIR,
    add_analysis_features,
    load_clean_listings,
    summarize_age_bands,
    summarize_daily_listings,
    summarize_manufacturers,
    summarize_mileage_bands,
    summarize_states,
    summarize_vehicle_types,
)
from src.data.ingestion import PROJECT_ROOT

FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"


def save_figure(filename: str) -> None:
    """Save the current Matplotlib figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / filename

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_price_distribution(listings: pd.DataFrame) -> None:
    """Plot the asking-price distribution for mainstream listings."""
    price_limit = listings["price"].quantile(0.99)
    filtered = listings.loc[listings["price"] <= price_limit]

    plt.figure(figsize=(9, 5))
    plt.hist(filtered["price"], bins=50)
    plt.title("Distribution of Vehicle Asking Prices")
    plt.xlabel("Asking Price ($)")
    plt.ylabel("Number of Listings")

    save_figure("price_distribution.png")


def plot_top_manufacturers(listings: pd.DataFrame) -> None:
    """Plot listing counts for the largest manufacturers."""
    summary = summarize_manufacturers(
        listings,
        minimum_listings=500,
    ).head(15)

    summary = summary.sort_values("listing_count")

    plt.figure(figsize=(9, 6))
    plt.barh(summary["manufacturer"], summary["listing_count"])
    plt.title("Top 15 Manufacturers by Listing Volume")
    plt.xlabel("Number of Listings")
    plt.ylabel("Manufacturer")

    save_figure("top_manufacturers.png")


def plot_price_by_vehicle_age(listings: pd.DataFrame) -> None:
    """Plot median asking price by vehicle-age band."""
    summary = summarize_age_bands(listings)

    plt.figure(figsize=(8, 5))
    plt.bar(
        summary["vehicle_age_band"].astype("string"),
        summary["median_price"],
    )
    plt.title("Median Asking Price by Vehicle Age")
    plt.xlabel("Vehicle Age")
    plt.ylabel("Median Asking Price ($)")
    plt.xticks(rotation=20)

    save_figure("price_by_vehicle_age.png")


def plot_price_by_mileage(listings: pd.DataFrame) -> None:
    """Plot median asking price by mileage band."""
    summary = summarize_mileage_bands(listings)

    plt.figure(figsize=(8, 5))
    plt.bar(
        summary["mileage_band"].astype("string"),
        summary["median_price"],
    )
    plt.title("Median Asking Price by Mileage")
    plt.xlabel("Mileage Band")
    plt.ylabel("Median Asking Price ($)")
    plt.xticks(rotation=20)

    save_figure("price_by_mileage.png")


def plot_vehicle_types(listings: pd.DataFrame) -> None:
    """Plot median asking price for common vehicle types."""
    summary = summarize_vehicle_types(listings)

    summary = summary.loc[summary["listing_count"] >= 1_000].sort_values("median_price")

    plt.figure(figsize=(9, 6))
    plt.barh(summary["type"], summary["median_price"])
    plt.title("Median Asking Price by Vehicle Type")
    plt.xlabel("Median Asking Price ($)")
    plt.ylabel("Vehicle Type")

    save_figure("price_by_vehicle_type.png")


def plot_state_prices(listings: pd.DataFrame) -> None:
    """Plot median asking prices for states with enough listings."""
    summary = summarize_states(
        listings,
        minimum_listings=1_000,
    )

    summary = summary.nlargest(15, "median_price").sort_values("median_price")

    plt.figure(figsize=(9, 6))
    plt.barh(summary["state"], summary["median_price"])
    plt.title("Highest Median Asking Prices by State")
    plt.xlabel("Median Asking Price ($)")
    plt.ylabel("State")

    save_figure("highest_price_states.png")


def plot_daily_listing_volume(listings: pd.DataFrame) -> None:
    """Plot daily listing activity over the dataset period."""
    summary = summarize_daily_listings(listings)

    plt.figure(figsize=(10, 5))
    plt.plot(
        summary["posting_day"],
        summary["listing_count"],
        marker="o",
    )
    plt.title("Daily Vehicle Listing Volume")
    plt.xlabel("Posting Date")
    plt.ylabel("Number of Listings")
    plt.xticks(rotation=30)

    save_figure("daily_listing_volume.png")


def main() -> None:
    """Generate the initial exploratory visualization set."""
    listings = load_clean_listings()
    analysis = add_analysis_features(listings)

    plot_price_distribution(analysis)
    plot_top_manufacturers(analysis)
    plot_price_by_vehicle_age(analysis)
    plot_price_by_mileage(analysis)
    plot_vehicle_types(analysis)
    plot_state_prices(analysis)
    plot_daily_listing_volume(analysis)

    print("Marketplace visualizations created successfully")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Summary tables available at: {REPORT_DATA_DIR}")


if __name__ == "__main__":
    main()
