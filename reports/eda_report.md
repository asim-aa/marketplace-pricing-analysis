# Exploratory Marketplace Analysis

## Dataset Scope

The analysis uses 361,956 cleaned Craigslist vehicle listings posted between April 4 and May 5, 2021. One row represents one advertised vehicle listing. Asking prices represent seller advertisements rather than verified transaction prices.

## Marketplace Overview

The cleaned marketplace has a median asking price of $15,990 and a mean asking price of approximately $19,274. The mean exceeds the median because the price distribution is right-skewed, with a smaller group of expensive vehicles extending the upper tail.

The median listing represents a model-year 2013 vehicle, approximately eight years old at posting, with 87,488 miles.

## Vehicle Age and Price

Median asking price declines substantially with vehicle age:

- 0–3 years: $31,600
- 4–7 years: $21,590
- 8–12 years: $10,995
- 13–20 years: $5,999
- 21+ years: $6,850

The modest increase in the oldest category may reflect classic, collectible, specialty, or selectively preserved vehicles. It should not be interpreted as evidence that vehicles generally appreciate after 20 years.

## Mileage and Price

Median asking price decreases consistently as mileage rises:

- 0–25,000 miles: $30,590
- 25,000–50,000 miles: $26,500
- 50,000–100,000 miles: $16,793
- 100,000–150,000 miles: $9,900
- 150,000–200,000 miles: $6,800
- 200,000+ miles: approximately $5,500

Mileage and vehicle age are related, so these unadjusted summaries do not isolate the independent effect of mileage.

## Vehicle Type

Pickup trucks and trucks occupy the highest-priced common market segments, while sedans and minivans have substantially lower median asking prices. This indicates that vehicle type is an essential feature for later pricing models.

The `other` and `unknown` categories should be interpreted cautiously because they combine heterogeneous or incompletely described listings.

## Manufacturer Composition

Ford and Chevrolet dominate listing volume, followed by Toyota, Honda, and Nissan. The concentration of listings among major manufacturers means that later model evaluation should report performance separately for high-volume and low-volume brands.

## Geographic Differences

Median asking prices vary across states, with Alaska, Montana, and Washington appearing among the higher-priced markets. These differences are descriptive and may partly reflect variation in vehicle type, vehicle age, mileage, manufacturer mix, and regional supply.

## Listing Activity Over Time

Daily listing counts generally increase toward the end of the observed period. However, the dataset covers only about one month, and the first and final dates are incomplete collection days. The apparent trend should therefore not be interpreted as long-term marketplace growth.

## Modeling Implications

The exploratory analysis supports including the following features in the pricing model:

- Vehicle age
- Odometer mileage
- Manufacturer
- Model
- Vehicle type
- State or region
- Condition
- Fuel type
- Transmission
- Drive type

The analysis also motivates segment-level error reporting because pricing behavior differs materially across vehicle ages, mileage bands, types, manufacturers, and geographic markets.

## Limitations

- Asking prices are not verified transaction prices.
- The dataset covers only April–May 2021.
- Repeated VINs suggest reposted or duplicated physical vehicles.
- Geographic comparisons are not adjusted for vehicle composition.
- Older and specialty vehicles may behave differently from mainstream used vehicles.
- Missing optional attributes were retained as `unknown`.
