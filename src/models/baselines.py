"""Simple baseline models for vehicle asking-price prediction."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.analysis.market_overview import add_analysis_features
from src.data.ingestion import PROJECT_ROOT

SPLIT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

TRAIN_PATH = SPLIT_DATA_DIR / "train.parquet"
CALIBRATION_PATH = SPLIT_DATA_DIR / "calibration.parquet"
TEST_PATH = SPLIT_DATA_DIR / "test.parquet"


@dataclass
class GlobalMedianBaseline:
    """Predict the same training-set median price for every listing."""

    median_price: float | None = None

    def fit(self, listings: pd.DataFrame) -> "GlobalMedianBaseline":
        """Learn the global median asking price."""
        if listings.empty:
            raise ValueError("Cannot fit the baseline on an empty dataset.")

        self.median_price = float(listings["price"].median())
        return self

    def predict(self, listings: pd.DataFrame) -> pd.Series:
        """Predict the learned median price for every row."""
        if self.median_price is None:
            raise ValueError("The baseline must be fitted before prediction.")

        return pd.Series(
            self.median_price,
            index=listings.index,
            name="prediction",
            dtype="float64",
        )


@dataclass
class SegmentMedianBaseline:
    """Predict stable hierarchical medians for vehicle segments."""

    minimum_detailed_count: int = 5
    minimum_manufacturer_age_count: int = 20

    global_median: float | None = None
    manufacturer_medians: pd.Series | None = None
    manufacturer_age_medians: pd.Series | None = None
    detailed_medians: pd.Series | None = None

    def fit(self, listings: pd.DataFrame) -> "SegmentMedianBaseline":
        """Learn stable training-only medians at several specificity levels."""
        if listings.empty:
            raise ValueError("Cannot fit the baseline on an empty dataset.")

        if self.minimum_detailed_count <= 0:
            raise ValueError("minimum_detailed_count must be greater than zero.")

        if self.minimum_manufacturer_age_count <= 0:
            raise ValueError(
                "minimum_manufacturer_age_count must be greater than zero."
            )

        training = add_analysis_features(listings)

        self.global_median = float(training["price"].median())

        self.manufacturer_medians = training.groupby(
            "manufacturer",
            observed=True,
        )["price"].median()

        manufacturer_age_stats = training.groupby(
            ["manufacturer", "vehicle_age_band"],
            observed=True,
        )["price"].agg(["median", "count"])

        self.manufacturer_age_medians = manufacturer_age_stats.loc[
            manufacturer_age_stats["count"] >= self.minimum_manufacturer_age_count,
            "median",
        ]

        detailed_stats = training.groupby(
            ["manufacturer", "model", "vehicle_age_band"],
            observed=True,
        )["price"].agg(["median", "count"])

        self.detailed_medians = detailed_stats.loc[
            detailed_stats["count"] >= self.minimum_detailed_count,
            "median",
        ]

        return self

    def predict(self, listings: pd.DataFrame) -> pd.Series:
        """Predict using stable segments with hierarchical fallbacks."""
        if (
            self.global_median is None
            or self.manufacturer_medians is None
            or self.manufacturer_age_medians is None
            or self.detailed_medians is None
        ):
            raise ValueError("The baseline must be fitted before prediction.")

        prediction_data = add_analysis_features(listings)

        predictions = pd.Series(
            index=prediction_data.index,
            dtype="float64",
            name="prediction",
        )

        detailed_keys = pd.MultiIndex.from_frame(
            prediction_data[
                [
                    "manufacturer",
                    "model",
                    "vehicle_age_band",
                ]
            ]
        )

        predictions[:] = self.detailed_medians.reindex(detailed_keys).to_numpy()

        missing = predictions.isna()

        if missing.any():
            manufacturer_age_keys = pd.MultiIndex.from_frame(
                prediction_data.loc[
                    missing,
                    [
                        "manufacturer",
                        "vehicle_age_band",
                    ],
                ]
            )

            predictions.loc[missing] = self.manufacturer_age_medians.reindex(
                manufacturer_age_keys
            ).to_numpy()

        missing = predictions.isna()

        if missing.any():
            predictions.loc[missing] = (
                prediction_data.loc[missing, "manufacturer"]
                .map(self.manufacturer_medians)
                .to_numpy()
            )

        return predictions.fillna(self.global_median)


def load_split(path: Path) -> pd.DataFrame:
    """Load one saved dataset split."""
    if not path.exists():
        raise FileNotFoundError(
            f"Split not found at {path}. "
            "Run `uv run python -m src.data.splitting` first."
        )

    return pd.read_parquet(path)


def main() -> None:
    """Fit both baselines and print example predictions."""
    train = load_split(TRAIN_PATH)
    test = load_split(TEST_PATH)

    global_model = GlobalMedianBaseline().fit(train)
    segment_model = SegmentMedianBaseline().fit(train)

    global_predictions = global_model.predict(test)
    segment_predictions = segment_model.predict(test)

    print("Baseline models fitted successfully")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Global training median: ${global_model.median_price:,.2f}")
    print()
    print("Segment rules:")
    print(f"Detailed group minimum: {segment_model.minimum_detailed_count} rows")
    print(
        f"Manufacturer-age minimum: {segment_model.minimum_manufacturer_age_count} rows"
    )
    print()
    print("Example predictions:")
    print(
        pd.DataFrame(
            {
                "actual_price": test["price"].head(),
                "global_prediction": global_predictions.head(),
                "segment_prediction": segment_predictions.head(),
            }
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
