# Uncertainty-Aware Marketplace Pricing & Opportunity Analysis

An end-to-end data science and business intelligence project that models used-vehicle asking prices, constructs calibrated prediction intervals, identifies listings outside expected market ranges, and communicates results through Tableau.

## Current status

Phase 1: project definition and dataset selection.

## Dataset

Craigslist Used Cars dataset by Austin Reese on Kaggle. Download `vehicles.csv` and place it in:

```text
data/raw/vehicles.csv
```

Do not commit the raw dataset to Git.

## Planned pipeline

```text
Raw listings
→ validation and cleaning
→ exploratory analysis
→ chronological train/calibration/test split
→ median and Ridge baselines
→ CatBoost model
→ conformal prediction intervals
→ opportunity classification
→ Tableau-ready scored dataset
```

## Setup

```bash
uv sync
```

## Quality checks

```bash
uv run ruff check .
uv run pytest
```
