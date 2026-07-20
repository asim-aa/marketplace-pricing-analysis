"""Analyze conformal coverage and marketplace pricing opportunities."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ingestion import PROJECT_ROOT
from src.models.baselines import TEST_PATH, load_split
from src.uncertainty.run_conformal import (
    CONFORMAL_METRICS_PATH,
    CONFORMAL_PREDICTIONS_PATH,
)

CONDITIONAL_COVERAGE_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "conditional_coverage_metrics.csv"
)

OPPORTUNITY_EXPORT_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "marketplace_opportunities.csv"
)

UNCERTAINTY_REPORT_PATH = PROJECT_ROOT / "reports" / "uncertainty_report.md"

NOMINAL_COVERAGE = 0.90

PRICE_BAND_LABELS = [
    "Under $5k",
    "$5k-$10k",
    "$10k-$20k",
    "$20k-$35k",
    "$35k-$50k",
    "$50k+",
]

AGE_BAND_LABELS = [
    "0-2 years",
    "3-5 years",
    "6-10 years",
    "11-15 years",
    "16-20 years",
    "21+ years",
]

MILEAGE_BAND_LABELS = [
    "Under 25k",
    "25k-50k",
    "50k-100k",
    "100k-150k",
    "150k-200k",
    "200k+",
]


def assign_price_band(prices: pd.Series) -> pd.Series:
    """Assign asking prices to ordered market bands."""
    return pd.cut(
        prices,
        bins=[
            float("-inf"),
            5_000,
            10_000,
            20_000,
            35_000,
            50_000,
            float("inf"),
        ],
        labels=PRICE_BAND_LABELS,
        right=False,
        ordered=True,
    )


def assign_age_band(vehicle_age: pd.Series) -> pd.Series:
    """Assign vehicle ages to ordered groups."""
    return pd.cut(
        vehicle_age,
        bins=[
            float("-inf"),
            3,
            6,
            11,
            16,
            21,
            float("inf"),
        ],
        labels=AGE_BAND_LABELS,
        right=False,
        ordered=True,
    )


def assign_mileage_band(odometer: pd.Series) -> pd.Series:
    """Assign odometer values to ordered mileage groups."""
    return pd.cut(
        odometer,
        bins=[
            float("-inf"),
            25_000,
            50_000,
            100_000,
            150_000,
            200_000,
            float("inf"),
        ],
        labels=MILEAGE_BAND_LABELS,
        right=False,
        ordered=True,
    )


def classify_opportunities(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    """Classify listings relative to their 90% prediction interval."""
    required_columns = {
        "price",
        "prediction",
        "lower_90",
        "upper_90",
        "width_90",
    }

    missing_columns = sorted(required_columns.difference(scored.columns))

    if missing_columns:
        missing_display = ", ".join(missing_columns)
        raise ValueError(
            f"Opportunity input is missing required columns: {missing_display}"
        )

    result = scored.copy()

    result["opportunity_90"] = np.select(
        [
            result["price"] < result["lower_90"],
            result["price"] > result["upper_90"],
        ],
        [
            "potentially_underpriced",
            "potentially_overpriced",
        ],
        default="market_consistent",
    )

    result["opportunity_gap_90"] = np.select(
        [
            result["opportunity_90"] == "potentially_underpriced",
            result["opportunity_90"] == "potentially_overpriced",
        ],
        [
            result["lower_90"] - result["price"],
            result["price"] - result["upper_90"],
        ],
        default=0.0,
    )

    result["relative_opportunity_gap_90"] = result["opportunity_gap_90"] / result[
        "prediction"
    ].clip(lower=500.0)

    result["covered_90"] = (result["price"] >= result["lower_90"]) & (
        result["price"] <= result["upper_90"]
    )

    result["interval_width_ratio_90"] = result["width_90"] / result["prediction"].clip(
        lower=500.0
    )

    return result


def add_segment_columns(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    """Add market segments used for conditional coverage analysis."""
    required_columns = {
        "price",
        "year",
        "odometer",
        "posting_date",
    }

    missing_columns = sorted(required_columns.difference(scored.columns))

    if missing_columns:
        missing_display = ", ".join(missing_columns)
        raise ValueError(
            f"Segment input is missing required columns: {missing_display}"
        )

    result = scored.copy()

    result["vehicle_age"] = (result["posting_date"].dt.year - result["year"]).clip(
        lower=0
    )

    result["price_band"] = assign_price_band(result["price"])

    result["vehicle_age_band"] = assign_age_band(result["vehicle_age"])

    result["mileage_band"] = assign_mileage_band(result["odometer"])

    return result


def summarize_conditional_coverage(
    scored: pd.DataFrame,
    segment_column: str,
    minimum_rows: int = 100,
) -> pd.DataFrame:
    """Summarize 90% interval behavior within one segment."""
    if segment_column not in scored.columns:
        raise ValueError(f"Segment column `{segment_column}` is missing.")

    if minimum_rows <= 0:
        raise ValueError("minimum_rows must be greater than zero.")

    required_columns = {
        "covered_90",
        "width_90",
        "absolute_error",
        "opportunity_90",
    }

    missing_columns = sorted(required_columns.difference(scored.columns))

    if missing_columns:
        missing_display = ", ".join(missing_columns)
        raise ValueError(
            f"Coverage input is missing required columns: {missing_display}"
        )

    grouped = (
        scored.groupby(
            segment_column,
            observed=True,
            dropna=False,
        )
        .agg(
            listing_count=("covered_90", "size"),
            empirical_coverage=(
                "covered_90",
                "mean",
            ),
            average_interval_width=(
                "width_90",
                "mean",
            ),
            median_interval_width=(
                "width_90",
                "median",
            ),
            mae=("absolute_error", "mean"),
            median_absolute_error=(
                "absolute_error",
                "median",
            ),
            potentially_underpriced_count=(
                "opportunity_90",
                lambda values: (values == "potentially_underpriced").sum(),
            ),
            potentially_overpriced_count=(
                "opportunity_90",
                lambda values: (values == "potentially_overpriced").sum(),
            ),
        )
        .reset_index()
    )

    grouped = grouped[grouped["listing_count"] >= minimum_rows].copy()

    grouped["nominal_coverage"] = NOMINAL_COVERAGE

    grouped["coverage_gap"] = (
        grouped["empirical_coverage"] - grouped["nominal_coverage"]
    )

    grouped["underpriced_rate"] = (
        grouped["potentially_underpriced_count"] / grouped["listing_count"]
    )

    grouped["overpriced_rate"] = (
        grouped["potentially_overpriced_count"] / grouped["listing_count"]
    )

    grouped.insert(
        0,
        "segment_type",
        segment_column,
    )

    grouped = grouped.rename(columns={segment_column: "segment_value"})

    return grouped.sort_values(
        ["empirical_coverage", "listing_count"],
        ascending=[True, False],
    ).reset_index(drop=True)


def build_conditional_coverage_table(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    """Combine conditional coverage results across market segments."""
    segment_columns = [
        "price_band",
        "vehicle_age_band",
        "mileage_band",
        "manufacturer",
        "type",
        "state",
    ]

    summaries = [
        summarize_conditional_coverage(
            scored=scored,
            segment_column=segment_column,
            minimum_rows=100,
        )
        for segment_column in segment_columns
    ]

    return pd.concat(
        summaries,
        ignore_index=True,
    )


def write_uncertainty_report(
    global_metrics: pd.DataFrame,
    conditional_metrics: pd.DataFrame,
    scored: pd.DataFrame,
    output_path: Path = UNCERTAINTY_REPORT_PATH,
) -> None:
    """Write the Phase 6 uncertainty and opportunity report."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    level_90 = global_metrics.loc[global_metrics["confidence_level"] == 0.90].iloc[0]

    opportunity_counts = (
        scored["opportunity_90"]
        .value_counts()
        .reindex(
            [
                "potentially_underpriced",
                "market_consistent",
                "potentially_overpriced",
            ],
            fill_value=0,
        )
    )

    eligible_segments = conditional_metrics[
        conditional_metrics["listing_count"] >= 200
    ].copy()

    worst_segments = eligible_segments.nsmallest(
        10,
        "empirical_coverage",
    )

    best_underpriced = scored[
        scored["opportunity_90"] == "potentially_underpriced"
    ].nlargest(
        10,
        "opportunity_gap_90",
    )

    lines = [
        "# Phase 6: Conformal Uncertainty Analysis",
        "",
        "## Method",
        "",
        (
            "The selected CatBoost configuration was retrained "
            "on a chronological proper-training subset. "
            "Absolute residuals from a later dedicated "
            "conformal-calibration subset were used to create "
            "split-conformal prediction intervals."
        ),
        "",
        "## Global 90% Interval",
        "",
        (f"- Calibration residual quantile: ${level_90['quantile']:,.2f}"),
        (f"- Empirical test coverage: {level_90['empirical_coverage']:.2%}"),
        (f"- Nominal coverage: {level_90['confidence_level']:.0%}"),
        (f"- Average interval width: ${level_90['average_interval_width']:,.2f}"),
        (f"- Median interval width: ${level_90['median_interval_width']:,.2f}"),
        "",
        "## Opportunity Classification",
        "",
        (
            "- Potentially underpriced: "
            f"{opportunity_counts['potentially_underpriced']:,}"
        ),
        (f"- Market-consistent: {opportunity_counts['market_consistent']:,}"),
        (f"- Potentially overpriced: {opportunity_counts['potentially_overpriced']:,}"),
        "",
        "The labels compare asking price with the model's "
        "90% prediction interval. They do not establish true "
        "fair value, transaction price, or investment return.",
        "",
        "## Lowest-Coverage Segments",
        "",
        worst_segments[
            [
                "segment_type",
                "segment_value",
                "listing_count",
                "empirical_coverage",
                "mae",
            ]
        ].to_markdown(index=False),
        "",
        "## Largest Potential Underpricing Gaps",
        "",
        best_underpriced[
            [
                "id",
                "manufacturer",
                "model",
                "price",
                "prediction",
                "lower_90",
                "opportunity_gap_90",
            ]
        ].to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        (
            "Observed coverage was modestly below nominal "
            "coverage on the later chronological test period. "
            "This suggests temporal distribution shift and "
            "means interval labels should be treated as "
            "decision-support signals rather than guarantees."
        ),
    ]

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Create conditional coverage and opportunity artifacts."""
    scored = pd.read_csv(CONFORMAL_PREDICTIONS_PATH)

    test = load_split(TEST_PATH)

    posting_dates = test[["id", "posting_date"]].copy()

    scored = scored.merge(
        posting_dates,
        on="id",
        how="left",
        validate="one_to_one",
    )

    if scored["posting_date"].isna().any():
        raise ValueError("Some scored rows are missing posting dates.")

    scored = add_segment_columns(scored)
    scored = classify_opportunities(scored)

    conditional_metrics = build_conditional_coverage_table(scored)

    global_metrics = pd.read_csv(CONFORMAL_METRICS_PATH)

    CONDITIONAL_COVERAGE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conditional_metrics.to_csv(
        CONDITIONAL_COVERAGE_PATH,
        index=False,
    )

    scored.to_csv(
        OPPORTUNITY_EXPORT_PATH,
        index=False,
    )

    write_uncertainty_report(
        global_metrics=global_metrics,
        conditional_metrics=conditional_metrics,
        scored=scored,
    )

    print("Phase 6 uncertainty analysis completed")
    print()
    print("Opportunity counts:")
    print(scored["opportunity_90"].value_counts().to_string())
    print()
    print("Lowest-coverage segments:")
    print(
        conditional_metrics[conditional_metrics["listing_count"] >= 200]
        .nsmallest(
            10,
            "empirical_coverage",
        )[
            [
                "segment_type",
                "segment_value",
                "listing_count",
                "empirical_coverage",
                "mae",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print()
    print(f"Conditional metrics: {CONDITIONAL_COVERAGE_PATH}")
    print(f"Opportunity export: {OPPORTUNITY_EXPORT_PATH}")
    print(f"Report: {UNCERTAINTY_REPORT_PATH}")


if __name__ == "__main__":
    main()
