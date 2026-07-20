"""Leakage-safe Ridge regression pipeline for vehicle asking prices."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.analysis.market_overview import add_analysis_features
from src.data.ingestion import PROJECT_ROOT
from src.models.baselines import TEST_PATH, TRAIN_PATH, load_split

MODEL_PATH = PROJECT_ROOT / "models" / "ridge_pipeline.joblib"

NUMERIC_FEATURES = [
    "year",
    "vehicle_age",
    "odometer",
    "mileage_per_year",
    "lat",
    "long",
]

CATEGORICAL_FEATURES = [
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
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class RidgePriceModel:
    """Ridge regression model with preprocessing and a log-price target."""

    def __init__(
        self,
        alpha: float = 10.0,
        minimum_category_frequency: int = 20,
    ) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be greater than zero.")

        if minimum_category_frequency <= 0:
            raise ValueError("minimum_category_frequency must be greater than zero.")

        self.alpha = alpha
        self.minimum_category_frequency = minimum_category_frequency
        self.pipeline = self._build_pipeline()
        self.is_fitted = False

    def _build_pipeline(self) -> TransformedTargetRegressor:
        """Create the full preprocessing and Ridge regression pipeline."""
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="unknown",
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=self.minimum_category_frequency,
                        sparse_output=True,
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    NUMERIC_FEATURES,
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    CATEGORICAL_FEATURES,
                ),
            ]
        )

        regression_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "ridge",
                    Ridge(
                        alpha=self.alpha,
                        solver="lsqr",
                    ),
                ),
            ]
        )

        return TransformedTargetRegressor(
            regressor=regression_pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )

    @staticmethod
    def prepare_features(listings: pd.DataFrame) -> pd.DataFrame:
        """Build analysis features and return the model feature matrix."""
        featured = add_analysis_features(listings)

        missing_columns = [
            column for column in MODEL_FEATURES if column not in featured.columns
        ]

        if missing_columns:
            missing_display = ", ".join(missing_columns)
            raise ValueError(
                f"Model input is missing required columns: {missing_display}"
            )

        return featured[MODEL_FEATURES].copy()

    def fit(self, listings: pd.DataFrame) -> "RidgePriceModel":
        """Fit preprocessing and Ridge regression using training data."""
        if listings.empty:
            raise ValueError("Cannot fit Ridge on an empty dataset.")

        features = self.prepare_features(listings)
        target = listings["price"].astype("float64")

        self.pipeline.fit(features, target)
        self.is_fitted = True

        return self

    def predict(self, listings: pd.DataFrame) -> pd.Series:
        """Predict asking prices and enforce the modeled price floor."""
        if not self.is_fitted:
            raise ValueError("The Ridge model must be fitted before prediction.")

        features = self.prepare_features(listings)
        predictions = self.pipeline.predict(features)

        predictions = np.maximum(predictions, 500.0)

        return pd.Series(
            predictions,
            index=listings.index,
            name="prediction",
            dtype="float64",
        )

    def save(self, path: Path = MODEL_PATH) -> None:
        """Save the fitted Ridge model."""
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted Ridge model.")

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


def main() -> None:
    """Fit, save, and demonstrate the Ridge price model."""
    train = load_split(TRAIN_PATH)
    test = load_split(TEST_PATH)

    model = RidgePriceModel().fit(train)
    predictions = model.predict(test)

    model.save()

    print("Ridge price model fitted successfully")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Numeric features: {len(NUMERIC_FEATURES)}")
    print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"Model saved to: {MODEL_PATH}")
    print()
    print("Example predictions:")
    print(
        pd.DataFrame(
            {
                "actual_price": test["price"].head(),
                "ridge_prediction": predictions.head(),
            }
        )
        .round(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
