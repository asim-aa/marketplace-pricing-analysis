# Baseline Model Results

## Evaluation Design

Models were fitted using the chronological training split and evaluated on the untouched chronological test split.

## Overall Metrics

|                |      mae |     rmse |   median_absolute_error |
|:---------------|---------:|---------:|------------------------:|
| global_median  | 10790.7  | 14497.2  |                  9778   |
| segment_median |  5036.32 |  9204.85 |                  2797.5 |

## Main Findings

- Lowest MAE: `segment_median`
- Lowest median absolute error: `segment_median`
- The global median baseline predicts one training-set median price for every listing.
- The segment median baseline uses training-only medians with hierarchical fallbacks.

## Error by Actual Price Band

| model          | price_band   |   listing_count |      mae |   median_absolute_error |   median_actual_price |   median_prediction |
|:---------------|:-------------|----------------:|---------:|------------------------:|----------------------:|--------------------:|
| global_median  | Under $5k    |            8834 | 13597.7  |                   13323 |                  3450 |             16773   |
| global_median  | $5k-$10k     |           10338 |  9186.08 |                    9273 |                  7500 |             16773   |
| global_median  | $10k-$20k    |           11183 |  2812.27 |                    2773 |                 14995 |             16773   |
| global_median  | $20k-$35k    |            7555 | 10426.1  |                   10222 |                 26995 |             16773   |
| global_median  | $35k-$50k    |            2648 | 24486.3  |                   23217 |                 39990 |             16773   |
| global_median  | $50k+        |            1188 | 50776.2  |                   45222 |                 61995 |             16773   |
| segment_median | Under $5k    |            8834 |  4091.08 |                    2000 |                  3450 |              4999   |
| segment_median | $5k-$10k     |           10338 |  2499.03 |                    1700 |                  7500 |              7799.5 |
| segment_median | $10k-$20k    |           11183 |  4149.61 |                    3256 |                 14995 |             13995   |
| segment_median | $20k-$35k    |            7555 |  5877.52 |                    4400 |                 26995 |             24929   |
| segment_median | $35k-$50k    |            2648 |  9543.08 |                    7400 |                 39990 |             34791   |
| segment_median | $50k+        |            1188 | 27096.6  |                   21433 |                 61995 |             38998   |

## Interpretation

These baselines establish the minimum performance that later models must exceed. A more complex model is only useful if it improves meaningfully on these simple, leakage-safe benchmarks.
