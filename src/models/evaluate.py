"""Evaluate vehicle pricing predictions with regression metrics."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)

from src.data.ingestion import PROJECT_ROOT
from src.models.baselines import (
    TEST_PATH,
    TRAIN_PATH,
    GlobalMedianBaseline,
    SegmentMedianBaseline,
    load_split,
)
from src.models.ridge import RidgePriceModel

REPORT_PATH = PROJECT_ROOT / "reports" / "baseline_results.md"
TABLE_PATH = PROJECT_ROOT / "reports" / "tables" / "baseline_metrics.csv"


def calculate_regression_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> pd.Series:
    """Calculate the main regression evaluation metrics."""
    return pd.Series(
        {
            "mae": mean_absolute_error(actual, predicted),
            "rmse": np.sqrt(mean_squared_error(actual, predicted)),
            "median_absolute_error": median_absolute_error(
                actual,
                predicted,
            ),
        }
    )


def add_price_bands(listings: pd.DataFrame) -> pd.DataFrame:
    """Assign each listing to a price band using actual asking price."""
    evaluated = listings.copy()

    evaluated["price_band"] = pd.cut(
        evaluated["price"],
        bins=[
            0,
            5_000,
            10_000,
            20_000,
            35_000,
            50_000,
            float("inf"),
        ],
        labels=[
            "Under $5k",
            "$5k-$10k",
            "$10k-$20k",
            "$20k-$35k",
            "$35k-$50k",
            "$50k+",
        ],
    )

    return evaluated


def evaluate_by_price_band(
    actual: pd.Series,
    predicted: pd.Series,
) -> pd.DataFrame:
    """Calculate prediction errors within actual-price bands."""
    evaluation = pd.DataFrame(
        {
            "price": actual,
            "prediction": predicted,
        }
    )

    evaluation = add_price_bands(evaluation)
    evaluation["absolute_error"] = (
        evaluation["price"] - evaluation["prediction"]
    ).abs()

    return (
        evaluation.groupby("price_band", observed=True)
        .agg(
            listing_count=("price", "count"),
            mae=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            median_actual_price=("price", "median"),
            median_prediction=("prediction", "median"),
        )
        .reset_index()
    )


def evaluate_model(
    model_name: str,
    actual: pd.Series,
    predicted: pd.Series,
) -> tuple[pd.Series, pd.DataFrame]:
    """Evaluate one model overall and by price band."""
    metrics = calculate_regression_metrics(actual, predicted)
    metrics.name = model_name

    price_band_metrics = evaluate_by_price_band(actual, predicted)
    price_band_metrics.insert(0, "model", model_name)

    return metrics, price_band_metrics


def write_baseline_report(
    metrics: pd.DataFrame,
    price_band_metrics: pd.DataFrame,
    output_path: Path = REPORT_PATH,
) -> None:
    """Write baseline evaluation results as Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_mae_model = metrics["mae"].idxmin()
    best_median_model = metrics["median_absolute_error"].idxmin()

    lines = [
        "# Baseline Model Results",
        "",
        "## Evaluation Design",
        "",
        (
            "Models were fitted using the chronological training split and "
            "evaluated on the untouched chronological test split."
        ),
        "",
        "## Overall Metrics",
        "",
        metrics.round(2).to_markdown(),
        "",
        "## Main Findings",
        "",
        f"- Lowest MAE: `{best_mae_model}`",
        f"- Lowest median absolute error: `{best_median_model}`",
        (
            "- The global median baseline predicts one training-set median "
            "price for every listing."
        ),
        (
            "- The segment median baseline uses training-only medians with "
            "hierarchical fallbacks."
        ),
        "",
        "## Error by Actual Price Band",
        "",
        price_band_metrics.round(2).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        (
            "These baselines establish the minimum performance that later "
            "models must exceed. A more complex model is only useful if it "
            "improves meaningfully on these simple, leakage-safe benchmarks."
        ),
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Fit, evaluate, save, and report both baseline models."""
    train = load_split(TRAIN_PATH)
    test = load_split(TEST_PATH)

    models = {
        "global_median": GlobalMedianBaseline().fit(train),
        "segment_median": SegmentMedianBaseline().fit(train),
        "ridge": RidgePriceModel().fit(train),
    }
    overall_metrics = []
    price_band_results = []

    for model_name, model in models.items():
        predictions = model.predict(test)

        metrics, band_metrics = evaluate_model(
            model_name=model_name,
            actual=test["price"],
            predicted=predictions,
        )

        overall_metrics.append(metrics)
        price_band_results.append(band_metrics)

    metrics_table = pd.DataFrame(overall_metrics)
    price_band_table = pd.concat(
        price_band_results,
        ignore_index=True,
    )

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(TABLE_PATH)
    price_band_table.to_csv(
        TABLE_PATH.with_name("baseline_price_band_metrics.csv"),
        index=False,
    )

    write_baseline_report(
        metrics=metrics_table,
        price_band_metrics=price_band_table,
    )

    print("Baseline evaluation completed successfully")
    print()
    print(metrics_table.round(2))
    print()
    print(f"Metrics saved to: {TABLE_PATH}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
