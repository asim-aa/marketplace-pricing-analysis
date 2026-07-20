# Phase 6: Conformal Uncertainty Analysis

## Method

The selected CatBoost configuration was retrained on a chronological proper-training subset. Absolute residuals from a later dedicated conformal-calibration subset were used to create split-conformal prediction intervals.

## Global 90% Interval

- Calibration residual quantile: $6,981.67
- Empirical test coverage: 87.88%
- Nominal coverage: 90%
- Average interval width: $13,204.86
- Median interval width: $13,963.33

## Opportunity Classification

- Potentially underpriced: 2,040
- Market-consistent: 36,686
- Potentially overpriced: 3,020

The labels compare asking price with the model's 90% prediction interval. They do not establish true fair value, transaction price, or investment return.

## Lowest-Coverage Segments

| segment_type     | segment_value   |   listing_count |   empirical_coverage |      mae |
|:-----------------|:----------------|----------------:|---------------------:|---------:|
| price_band       | $50k+           |            1210 |             0.234711 | 20472.3  |
| manufacturer     | porsche         |             210 |             0.547619 | 14830.9  |
| price_band       | $35k-$50k       |            2678 |             0.602315 |  7050.96 |
| manufacturer     | rover           |             239 |             0.644351 |  9208.11 |
| vehicle_age_band | 0-2 years       |            3254 |             0.691457 |  7581.94 |
| mileage_band     | Under 25k       |            4780 |             0.706067 |  7465.45 |
| manufacturer     | ram             |            1410 |             0.770213 |  5774.8  |
| mileage_band     | 25k-50k         |            4795 |             0.792492 |  5403.8  |
| state            | hi              |             353 |             0.793201 |  5905.04 |
| vehicle_age_band | 3-5 years       |            7137 |             0.802718 |  5099.25 |

## Largest Potential Underpricing Gaps

|         id | manufacturer   | model                   |   price |   prediction |   lower_90 |   opportunity_gap_90 |
|-----------:|:---------------|:------------------------|--------:|-------------:|-----------:|---------------------:|
| 7316324147 | ford           | f-250 super duty lariat |    1008 |      76346.4 |    69364.7 |              68356.7 |
| 7316794426 | ferrari        | california t            |    2034 |      75722.8 |    68741.1 |              66707.1 |
| 7316794326 | ferrari        | california t            |    2034 |      74110.2 |    67128.5 |              65094.5 |
| 7316732946 | ford           | f-350 super duty xlt    |     883 |      72442.2 |    65460.5 |              64577.5 |
| 7316840776 | chevrolet      | silverado 2500 lt       |     771 |      71997.8 |    65016.1 |              64245.1 |
| 7316778354 | ford           | f-350 super duty lariat |     980 |      70356.6 |    63375   |              62395   |
| 7316312309 | gmc            | sierra 1500 denali      |    1030 |      69482.9 |    62501.3 |              61471.3 |
| 7316573882 | chevrolet      | silverado 2500 ltz      |     889 |      68742.9 |    61761.3 |              60872.3 |
| 7316579613 | chevrolet      | silverado 2500 lt       |     785 |      66406.9 |    59425.3 |              58640.3 |
| 7316745372 | ram            | 2500 laramie            |     830 |      66012.2 |    59030.5 |              58200.5 |

## Interpretation

Observed coverage was modestly below nominal coverage on the later chronological test period. This suggests temporal distribution shift and means interval labels should be treated as decision-support signals rather than guarantees.