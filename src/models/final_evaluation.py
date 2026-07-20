"""Final test-set evaluation for the selected CatBoost pricing model."""

import pandas as pd

from src.data.ingestion import PROJECT_ROOT
from src.models.baselines import (
    TEST_PATH,
    TRAIN_PATH,
    GlobalMedianBaseline,
    SegmentMedianBaseline,
    load_split,
)
from src.models.catboost_model import MODEL_PATH, CatBoostPriceModel
from src.models.evaluate import calculate_regression_metrics

FINAL_METRICS_PATH = PROJECT_ROOT / "reports" / "tables" / "final_model_metrics.csv"

FINAL_PRICE_BAND_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "final_model_price_band_metrics.csv"
)

TEST_PREDICTIONS_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "catboost_test_predictions.csv"
)

PRICE_BAND_ORDER = [
    "Under $5k",
    "$5k-$10k",
    "$10k-$20k",
    "$20k-$35k",
    "$35k-$50k",
    "$50k+",
]


def evaluate_model(
    name: str,
    actual: pd.Series,
    predicted: pd.Series,
) -> pd.Series:
    """Calculate overall metrics for one model."""
    metrics = calculate_regression_metrics(
        actual=actual,
        predicted=predicted,
    )
    metrics.name = name
    return metrics


def assign_price_band(actual: pd.Series) -> pd.Series:
    """Assign actual asking prices to ordered evaluation bands."""
    return pd.cut(
        actual,
        bins=[
            float("-inf"),
            5_000,
            10_000,
            20_000,
            35_000,
            50_000,
            float("inf"),
        ],
        labels=PRICE_BAND_ORDER,
        right=False,
        ordered=True,
    )


def calculate_price_band_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> pd.DataFrame:
    """Calculate regression metrics within actual-price bands."""
    evaluation = pd.DataFrame(
        {
            "actual": actual.astype("float64"),
            "predicted": predicted.astype("float64"),
        }
    )

    evaluation["absolute_error"] = (
        evaluation["actual"] - evaluation["predicted"]
    ).abs()

    evaluation["price_band"] = assign_price_band(evaluation["actual"])

    grouped = (
        evaluation.groupby(
            "price_band",
            observed=False,
        )
        .agg(
            listing_count=("actual", "size"),
            mae=("absolute_error", "mean"),
            median_absolute_error=(
                "absolute_error",
                "median",
            ),
            median_actual_price=("actual", "median"),
            median_prediction=("predicted", "median"),
        )
        .reset_index()
    )

    return grouped


def main() -> None:
    """Evaluate selected CatBoost and benchmarks on final test data."""
    train = load_split(TRAIN_PATH)
    test = load_split(TEST_PATH)

    global_model = GlobalMedianBaseline().fit(train)
    segment_model = SegmentMedianBaseline().fit(train)
    catboost_model = CatBoostPriceModel().load(MODEL_PATH)

    predictions = {
        "global_median": global_model.predict(test),
        "segment_median": segment_model.predict(test),
        "catboost": catboost_model.predict(test),
    }

    overall_metrics = pd.DataFrame(
        [
            evaluate_model(
                name=model_name,
                actual=test["price"],
                predicted=model_predictions,
            )
            for model_name, model_predictions in predictions.items()
        ]
    )

    price_band_results = []

    for model_name, model_predictions in predictions.items():
        band_metrics = calculate_price_band_metrics(
            actual=test["price"],
            predicted=model_predictions,
        )
        band_metrics.insert(0, "model", model_name)
        price_band_results.append(band_metrics)

    price_band_metrics = pd.concat(
        price_band_results,
        ignore_index=True,
    )

    prediction_output = test[
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

    prediction_output["catboost_prediction"] = predictions["catboost"]

    prediction_output["residual"] = (
        prediction_output["price"] - prediction_output["catboost_prediction"]
    )

    prediction_output["absolute_error"] = prediction_output["residual"].abs()

    FINAL_METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_metrics.to_csv(FINAL_METRICS_PATH)

    price_band_metrics.to_csv(
        FINAL_PRICE_BAND_PATH,
        index=False,
    )

    prediction_output.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
    )

    print("Final test evaluation completed successfully")
    print()
    print(overall_metrics.round(2))
    print()
    print(f"Metrics saved to: {FINAL_METRICS_PATH}")
    print(f"Price-band metrics saved to: {FINAL_PRICE_BAND_PATH}")
    print(f"Predictions saved to: {TEST_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
