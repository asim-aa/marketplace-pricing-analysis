"""Train and evaluate split-conformal prediction intervals."""

import pandas as pd

from src.data.ingestion import PROJECT_ROOT
from src.models.baselines import TEST_PATH, TRAIN_PATH, load_split
from src.models.catboost_model import CatBoostPriceModel
from src.uncertainty.conformal import (
    build_prediction_intervals,
    calculate_absolute_residuals,
    conformal_quantile,
    empirical_coverage,
    split_proper_training_and_conformal,
)

CONFORMAL_MODEL_PATH = PROJECT_ROOT / "models" / "conformal_pricing_model.cbm"

CONFORMAL_METRICS_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "conformal_interval_metrics.csv"
)

CONFORMAL_PREDICTIONS_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "conformal_test_predictions.csv"
)

CONFIDENCE_LEVELS = [0.80, 0.90, 0.95]


def train_conformal_model(
    proper_training: pd.DataFrame,
) -> CatBoostPriceModel:
    """Train the fixed Phase 5 CatBoost configuration."""
    model = CatBoostPriceModel(
        iterations=998,
        depth=8,
        learning_rate=0.10,
        verbose=100,
    )

    model.fit(
        train_listings=proper_training,
        validation_listings=None,
        early_stopping_rounds=None,
    )

    return model


def main() -> None:
    """Generate and evaluate conformal prediction intervals."""
    original_training = load_split(TRAIN_PATH)
    test = load_split(TEST_PATH)

    proper_training, conformal_calibration = split_proper_training_and_conformal(
        original_training,
        conformal_fraction=0.20,
    )

    print("Conformal data split")
    print(f"Proper training rows: {len(proper_training):,}")
    print(f"Conformal calibration rows: {len(conformal_calibration):,}")
    print(f"Test rows: {len(test):,}")
    print()

    model = train_conformal_model(proper_training)
    model.save(CONFORMAL_MODEL_PATH)

    calibration_predictions = model.predict(conformal_calibration)

    calibration_residuals = calculate_absolute_residuals(
        actual=conformal_calibration["price"],
        predicted=calibration_predictions,
    )

    test_predictions = model.predict(test)

    output = test[
        [
            "id",
            "price",
            "year",
            "manufacturer",
            "model",
            "odometer",
            "type",
            "state",
        ]
    ].copy()

    output["prediction"] = test_predictions
    output["residual"] = output["price"] - output["prediction"]
    output["absolute_error"] = output["residual"].abs()

    metric_rows = []

    for confidence_level in CONFIDENCE_LEVELS:
        quantile = conformal_quantile(
            absolute_residuals=calibration_residuals,
            confidence_level=confidence_level,
        )

        intervals = build_prediction_intervals(
            predictions=test_predictions,
            quantile=quantile,
            minimum_price=500.0,
        )

        level_label = int(confidence_level * 100)

        output[f"lower_{level_label}"] = intervals.lower_bound
        output[f"upper_{level_label}"] = intervals.upper_bound
        output[f"width_{level_label}"] = intervals.interval_width

        coverage = empirical_coverage(
            actual=test["price"],
            lower_bound=intervals.lower_bound,
            upper_bound=intervals.upper_bound,
        )

        metric_rows.append(
            {
                "confidence_level": confidence_level,
                "quantile": quantile,
                "empirical_coverage": coverage,
                "average_interval_width": (intervals.interval_width.mean()),
                "median_interval_width": (intervals.interval_width.median()),
                "test_rows": len(test),
                "calibration_rows": len(conformal_calibration),
            }
        )

    metrics = pd.DataFrame(metric_rows)

    CONFORMAL_METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        CONFORMAL_METRICS_PATH,
        index=False,
    )

    output.to_csv(
        CONFORMAL_PREDICTIONS_PATH,
        index=False,
    )

    print()
    print("Conformal interval evaluation complete")
    print()
    print(metrics.round(4).to_string(index=False))
    print()
    print(f"Model saved to: {CONFORMAL_MODEL_PATH}")
    print(f"Metrics saved to: {CONFORMAL_METRICS_PATH}")
    print(f"Predictions saved to: {CONFORMAL_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
