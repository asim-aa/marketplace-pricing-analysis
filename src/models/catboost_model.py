"""CatBoost regression model for vehicle asking-price prediction."""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from src.data.ingestion import PROJECT_ROOT
from src.features.build_features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    build_model_features,
    build_target,
)

MODEL_PATH = PROJECT_ROOT / "models" / "pricing_model.cbm"


class CatBoostPriceModel:
    """Nonlinear vehicle-price model with native categorical handling."""

    def __init__(
        self,
        iterations: int = 1_000,
        depth: int = 8,
        learning_rate: float = 0.05,
        loss_function: str = "MAE",
        random_seed: int = 42,
        verbose: int | bool = 100,
    ) -> None:
        if iterations <= 0:
            raise ValueError("iterations must be greater than zero.")

        if depth <= 0:
            raise ValueError("depth must be greater than zero.")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")

        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.random_seed = random_seed
        self.verbose = verbose

        self.model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function=loss_function,
            eval_metric="MAE",
            random_seed=random_seed,
            verbose=verbose,
            allow_writing_files=False,
        )

        self.is_fitted = False

    @staticmethod
    def _categorical_indices() -> list[int]:
        """Return the positions of categorical columns in the feature table."""
        return [MODEL_FEATURES.index(column) for column in CATEGORICAL_FEATURES]

    @staticmethod
    def _prepare_pool(
        listings: pd.DataFrame,
        include_target: bool,
    ) -> Pool:
        """Convert listings into a CatBoost Pool."""
        features = build_model_features(listings)

        categorical_indices = CatBoostPriceModel._categorical_indices()

        if include_target:
            target = build_target(listings)

            return Pool(
                data=features,
                label=target,
                cat_features=categorical_indices,
                feature_names=MODEL_FEATURES,
            )

        return Pool(
            data=features,
            cat_features=categorical_indices,
            feature_names=MODEL_FEATURES,
        )

    def fit(
        self,
        train_listings: pd.DataFrame,
        validation_listings: pd.DataFrame | None = None,
        early_stopping_rounds: int | None = 100,
    ) -> "CatBoostPriceModel":
        """Fit CatBoost using training data and optional validation data."""
        if train_listings.empty:
            raise ValueError("Cannot fit CatBoost on an empty dataset.")

        train_pool = self._prepare_pool(
            train_listings,
            include_target=True,
        )

        evaluation_pool = None

        if validation_listings is not None:
            if validation_listings.empty:
                raise ValueError("Validation data cannot be empty.")

            evaluation_pool = self._prepare_pool(
                validation_listings,
                include_target=True,
            )

        self.model.fit(
            train_pool,
            eval_set=evaluation_pool,
            early_stopping_rounds=(
                early_stopping_rounds if evaluation_pool is not None else None
            ),
            use_best_model=evaluation_pool is not None,
        )

        self.is_fitted = True
        return self

    def predict(self, listings: pd.DataFrame) -> pd.Series:
        """Predict vehicle asking prices."""
        if not self.is_fitted:
            raise ValueError("The CatBoost model must be fitted before prediction.")

        prediction_pool = self._prepare_pool(
            listings,
            include_target=False,
        )

        predictions = self.model.predict(prediction_pool)
        predictions = np.maximum(predictions, 500.0)

        return pd.Series(
            predictions,
            index=listings.index,
            name="prediction",
            dtype="float64",
        )

    def feature_importance(self) -> pd.DataFrame:
        """Return sorted CatBoost feature importance values."""
        if not self.is_fitted:
            raise ValueError("The CatBoost model must be fitted before inspection.")

        importance = pd.DataFrame(
            {
                "feature": MODEL_FEATURES,
                "importance": self.model.get_feature_importance(),
            }
        )

        return importance.sort_values(
            "importance",
            ascending=False,
        ).reset_index(drop=True)

    def save(self, path: Path = MODEL_PATH) -> None:
        """Save the fitted CatBoost model."""
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted CatBoost model.")

        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)

    def load(self, path: Path = MODEL_PATH) -> "CatBoostPriceModel":
        """Load a saved CatBoost model."""
        if not path.exists():
            raise FileNotFoundError(f"CatBoost model not found at {path}.")

        self.model.load_model(path)
        self.is_fitted = True

        return self
