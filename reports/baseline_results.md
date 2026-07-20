# Baseline Model Results

## Evaluation Design

Models were fitted using the chronological training split and evaluated on the untouched chronological test split.

## Overall Metrics

|                |      mae |     rmse |   median_absolute_error |
|:---------------|---------:|---------:|------------------------:|
| global_median  | 10790.7  | 14497.2  |                 9778    |
| segment_median |  5036.32 |  9204.85 |                 2797.5  |
| ridge          |  5164.4  | 47340.3  |                 2539.18 |

## Main Findings

- Lowest MAE: `segment_median`
- Lowest median absolute error: `ridge`
- The global median baseline predicts one training-set median price for every listing.
- The segment median baseline uses training-only medians with hierarchical fallbacks.

## Error by Actual Price Band

| model          | price_band   |   listing_count |      mae |   median_absolute_error |   median_actual_price |   median_prediction |
|:---------------|:-------------|----------------:|---------:|------------------------:|----------------------:|--------------------:|
| global_median  | Under $5k    |            8834 | 13597.7  |                13323    |                  3450 |            16773    |
| global_median  | $5k-$10k     |           10338 |  9186.08 |                 9273    |                  7500 |            16773    |
| global_median  | $10k-$20k    |           11183 |  2812.27 |                 2773    |                 14995 |            16773    |
| global_median  | $20k-$35k    |            7555 | 10426.1  |                10222    |                 26995 |            16773    |
| global_median  | $35k-$50k    |            2648 | 24486.3  |                23217    |                 39990 |            16773    |
| global_median  | $50k+        |            1188 | 50776.2  |                45222    |                 61995 |            16773    |
| segment_median | Under $5k    |            8834 |  4091.08 |                 2000    |                  3450 |             4999    |
| segment_median | $5k-$10k     |           10338 |  2499.03 |                 1700    |                  7500 |             7799.5  |
| segment_median | $10k-$20k    |           11183 |  4149.61 |                 3256    |                 14995 |            13995    |
| segment_median | $20k-$35k    |            7555 |  5877.52 |                 4400    |                 26995 |            24929    |
| segment_median | $35k-$50k    |            2648 |  9543.08 |                 7400    |                 39990 |            34791    |
| segment_median | $50k+        |            1188 | 27096.6  |                21433    |                 61995 |            38998    |
| ridge          | Under $5k    |            8834 |  3462.98 |                 1580.3  |                  3450 |             4988.97 |
| ridge          | $5k-$10k     |           10338 |  2107.36 |                 1426.02 |                  7500 |             7811.77 |
| ridge          | $10k-$20k    |           11183 |  4193.25 |                 2661.06 |                 14995 |            13162.4  |
| ridge          | $20k-$35k    |            7555 |  6677.36 |                 5723.55 |                 26995 |            21776.4  |
| ridge          | $35k-$50k    |            2648 | 12482.6  |                11514.4  |                 39990 |            31140    |
| ridge          | $50k+        |            1188 | 27626.9  |                22580.6  |                 61995 |            40165.8  |

## Interpretation

These baselines establish the minimum performance that later models must exceed. A more complex model is only useful if it improves meaningfully on these simple, leakage-safe benchmarks.
