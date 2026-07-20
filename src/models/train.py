"""Select and train the primary CatBoost vehicle-pricing model."""

from pathlib import Path

import pandas as pd

from src.data.ingestion import PROJECT_ROOT
from src.models.baselines import (
    CALIBRATION_PATH,
    TRAIN_PATH,
    load_split,
)
from src.models.catboost_model import (
    MODEL_PATH,
    CatBoostPriceModel,
)
from src.models.evaluate import calculate_regression_metrics

SELECTION_RESULTS_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "catboost_calibration_results.csv"
)

FEATURE_IMPORTANCE_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "catboost_feature_importance.csv"
)

CANDIDATE_CONFIGURATIONS = [
    {
        "name": "depth6_lr005",
        "iterations": 1_500,
        "depth": 6,
        "learning_rate": 0.05,
    },
    {
        "name": "depth8_lr005",
        "iterations": 1_500,
        "depth": 8,
        "learning_rate": 0.05,
    },
    {
        "name": "depth8_lr010",
        "iterations": 1_000,
        "depth": 8,
        "learning_rate": 0.10,
    },
]


def evaluate_candidate(
    configuration: dict[str, int | float | str],
    train: pd.DataFrame,
    calibration: pd.DataFrame,
) -> tuple[CatBoostPriceModel, pd.Series]:
    """Fit one candidate and evaluate it on calibration data."""
    model = CatBoostPriceModel(
        iterations=int(configuration["iterations"]),
        depth=int(configuration["depth"]),
        learning_rate=float(configuration["learning_rate"]),
        verbose=100,
    )

    model.fit(
        train_listings=train,
        validation_listings=calibration,
        early_stopping_rounds=100,
    )

    predictions = model.predict(calibration)

    metrics = calculate_regression_metrics(
        actual=calibration["price"],
        predicted=predictions,
    )

    metrics["best_iteration"] = model.model.get_best_iteration()
    metrics["tree_count"] = model.model.tree_count_

    return model, metrics


def select_best_model(
    results: pd.DataFrame,
) -> str:
    """Select the candidate with the lowest calibration MAE."""
    if results.empty:
        raise ValueError("No CatBoost candidate results were produced.")

    return str(results["mae"].idxmin())


def save_selection_results(
    results: pd.DataFrame,
    output_path: Path = SELECTION_RESULTS_PATH,
) -> None:
    """Save candidate calibration metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path)


def main() -> None:
    """Compare CatBoost candidates and save the best fitted model."""
    train = load_split(TRAIN_PATH)
    calibration = load_split(CALIBRATION_PATH)

    candidate_models: dict[str, CatBoostPriceModel] = {}
    candidate_metrics: list[pd.Series] = []

    for configuration in CANDIDATE_CONFIGURATIONS:
        candidate_name = str(configuration["name"])

        print()
        print("=" * 70)
        print(f"Training candidate: {candidate_name}")
        print("=" * 70)

        model, metrics = evaluate_candidate(
            configuration=configuration,
            train=train,
            calibration=calibration,
        )

        metrics.name = candidate_name

        candidate_models[candidate_name] = model
        candidate_metrics.append(metrics)

        print()
        print(metrics.round(2))

    results = pd.DataFrame(candidate_metrics)
    best_model_name = select_best_model(results)
    best_model = candidate_models[best_model_name]

    save_selection_results(results)
    best_model.save(MODEL_PATH)

    feature_importance = best_model.feature_importance()
    FEATURE_IMPORTANCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    feature_importance.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("CatBoost model selection complete")
    print("=" * 70)
    print(results.round(2))
    print()
    print(f"Selected model: {best_model_name}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Results saved to: {SELECTION_RESULTS_PATH}")
    print(f"Feature importance saved to: {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
