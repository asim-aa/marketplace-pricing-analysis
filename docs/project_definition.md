# Project Definition

## Project title

Uncertainty-Aware Marketplace Pricing & Opportunity Analysis

## Business problem

Used-vehicle marketplaces contain listings with large price differences caused by vehicle age, mileage, make, model, condition, seller type, geography, timing, missing information, and data-quality problems. A raw asking price alone does not tell a shopper or analyst whether a listing is broadly consistent with comparable listings.

This project builds a reproducible analytical pipeline that estimates a market-consistent asking price for each vehicle listing, quantifies uncertainty around that estimate, and flags listings whose asking prices fall outside the calibrated expected range.

## Unit of analysis

One row represents one Craigslist vehicle listing at the time it was posted.

## Prediction target

`price`: the listing's advertised asking price in U.S. dollars.

The target is an asking price, not a completed-sale price. Therefore, model outputs must be described as estimated market-consistent asking prices rather than guaranteed fair values or transaction values.

## Intended users

- Marketplace analyst studying used-vehicle pricing patterns
- Shopper or sourcing analyst screening listings for further investigation
- Business stakeholder evaluating market segments and model reliability

## Prediction timing

Predictions are generated using only information available when the listing is posted. Post-listing outcomes, future market aggregates, and target-derived features calculated from calibration or test rows are prohibited.

## Decision categories

- `Potentially Underpriced`: asking price is below the calibrated lower prediction bound
- `Within Expected Range`: asking price lies between the lower and upper bounds
- `Potentially Overpriced`: asking price is above the calibrated upper prediction bound

These labels are screening signals, not proof of listing quality or realized profit.

## Core analytical questions

1. Which attributes are associated with used-vehicle asking prices?
2. Does CatBoost improve prediction error over transparent baseline models?
3. How reliable are predictions across brands, price bands, locations, and seller types?
4. Do conformal prediction intervals achieve approximately their intended test-set coverage?
5. Which listings fall meaningfully outside their expected market range?

## Evaluation design

Listings will be sorted by `posting_date` and divided chronologically into:

- Training set: fit preprocessing, feature statistics, baselines, and CatBoost
- Calibration set: estimate conformal residual quantiles
- Test set: report final predictive performance, interval coverage, and segment errors

No random split will be used for the primary evaluation because a random split can allow later market conditions to inform predictions for earlier listings.

## Primary metrics

- Mean absolute error (MAE)
- Root mean squared error (RMSE)
- Median absolute error
- Improvement in MAE over the strongest baseline
- Prediction-interval empirical coverage
- Average and median prediction-interval width
- Segment-level MAE and coverage

## Baselines

1. Global training-set median price
2. Training-only segment median with hierarchical fallback
3. Ridge regression with a reproducible preprocessing pipeline

## Primary model

CatBoost regression, selected because the dataset contains many categorical variables, missing values, nonlinear relationships, and interactions.

## Leakage controls

- Never use `price` as an input feature.
- Never calculate price-derived aggregates from the full dataset.
- Fit encoders, imputers, scalers, segment medians, and regional price statistics on training data only.
- Use calibration residuals only to set conformal interval widths.
- Do not tune models using final test results.
- Exclude fields that directly identify or reproduce the target, including suspicious text or URL-derived price information.
- Review duplicate and near-duplicate listings before splitting to reduce cross-period duplication leakage.

## Known limitations

- Asking prices are not confirmed transaction prices.
- Listing condition and descriptions may be inaccurate or incomplete.
- Craigslist data is observational and may contain fraud, duplicates, placeholders, and extreme values.
- The historical dataset may not represent the current vehicle market.
- A low predicted price does not account for mechanical defects, title problems, scams, or inspection results.
- Prediction intervals quantify model residual uncertainty under the evaluation setup; they do not capture every real-world risk.

## Completion statement

One row represents one vehicle listing. Using information available when the listing is posted, the system estimates a market-consistent asking price and determines whether the listed price falls outside a calibrated expected range.
