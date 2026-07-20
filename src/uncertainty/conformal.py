"""Split-conformal prediction intervals for vehicle asking prices."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConformalInterval:
    """Container for calibrated prediction intervals."""

    prediction: pd.Series
    lower_bound: pd.Series
    upper_bound: pd.Series
    interval_width: pd.Series


def split_proper_training_and_conformal(
    training: pd.DataFrame,
    conformal_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create chronological proper-training and conformal subsets."""
    if training.empty:
        raise ValueError("Training data cannot be empty.")

    if not 0 < conformal_fraction < 1:
        raise ValueError("conformal_fraction must be between zero and one.")

    if "posting_date" not in training.columns:
        raise ValueError("Training data must contain posting_date.")

    ordered = training.sort_values(
        ["posting_date", "id"],
        kind="stable",
    ).reset_index(drop=True)

    split_index = int(np.floor(len(ordered) * (1 - conformal_fraction)))

    if split_index <= 0 or split_index >= len(ordered):
        raise ValueError(
            "Split would create an empty proper-training or conformal subset."
        )

    proper_training = ordered.iloc[:split_index].copy()
    conformal_calibration = ordered.iloc[split_index:].copy()

    return proper_training, conformal_calibration


def calculate_absolute_residuals(
    actual: pd.Series,
    predicted: pd.Series,
) -> pd.Series:
    """Return absolute calibration residuals."""
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted values must have equal length.")

    residuals = (actual.astype("float64") - predicted.astype("float64")).abs()

    if residuals.isna().any():
        raise ValueError("Calibration residuals contain missing values.")

    return residuals.rename("absolute_residual")


def conformal_quantile(
    absolute_residuals: pd.Series,
    confidence_level: float = 0.90,
) -> float:
    """Calculate the finite-sample split-conformal quantile."""
    if absolute_residuals.empty:
        raise ValueError("Calibration residuals cannot be empty.")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one.")

    residuals = absolute_residuals.to_numpy(
        dtype="float64",
    )

    if np.isnan(residuals).any():
        raise ValueError("Calibration residuals contain missing values.")

    sample_size = len(residuals)

    adjusted_level = np.ceil((sample_size + 1) * confidence_level) / sample_size

    adjusted_level = min(adjusted_level, 1.0)

    return float(
        np.quantile(
            residuals,
            adjusted_level,
            method="higher",
        )
    )


def build_prediction_intervals(
    predictions: pd.Series,
    quantile: float,
    minimum_price: float = 500.0,
) -> ConformalInterval:
    """Construct symmetric conformal prediction intervals."""
    if quantile < 0:
        raise ValueError("Conformal quantile cannot be negative.")

    if minimum_price < 0:
        raise ValueError("minimum_price cannot be negative.")

    prediction = predictions.astype("float64").rename("prediction")

    lower_bound = (
        (prediction - quantile).clip(lower=minimum_price).rename("lower_bound")
    )

    upper_bound = (prediction + quantile).rename("upper_bound")

    interval_width = (upper_bound - lower_bound).rename("interval_width")

    return ConformalInterval(
        prediction=prediction,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        interval_width=interval_width,
    )


def empirical_coverage(
    actual: pd.Series,
    lower_bound: pd.Series,
    upper_bound: pd.Series,
) -> float:
    """Calculate the fraction of outcomes inside their intervals."""
    if not (len(actual) == len(lower_bound) == len(upper_bound)):
        raise ValueError("Actual values and interval bounds must have equal length.")

    covered = (actual.astype("float64") >= lower_bound.astype("float64")) & (
        actual.astype("float64") <= upper_bound.astype("float64")
    )

    return float(covered.mean())
