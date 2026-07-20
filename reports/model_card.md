# Model Card: Uncertainty-Aware Marketplace Pricing Model

## 1. Model Overview

This project estimates a **market-consistent asking price** for used vehicle listings and identifies listings whose advertised prices appear unusually low or unusually high relative to the model's expected range.

The final system combines:

- A **CatBoost regression model** for point predictions
- **Split conformal prediction** for 90% uncertainty intervals
- Rule-based opportunity labels
- A Tableau dashboard for interactive analysis

The model was trained on structured Craigslist used-vehicle listing data.

The system predicts listing behavior, not verified vehicle value. It does not observe completed sale prices, professional appraisals, inspection results, or negotiation outcomes.

---

## 2. Intended Use

The model is intended for exploratory marketplace analysis and decision support.

Appropriate uses include:

- Estimating a market-consistent asking price for a used vehicle listing
- Identifying listings whose asking prices fall outside the model's expected range
- Prioritizing unusual listings for manual review
- Comparing pricing patterns by manufacturer, state, vehicle type, price band, confidence tier, and signal strength
- Demonstrating an end-to-end machine-learning workflow involving data engineering, regression, uncertainty estimation, automated testing, and business intelligence

A potentially underpriced signal means:

> The listing's asking price is lower than the model's uncertainty-adjusted expected range based on the available structured information.

It does **not** mean:

> The vehicle is definitely worth more than the asking price.

The intended workflow is:

1. The model identifies an unusual listing.
2. A person reviews the listing.
3. The vehicle, seller, title, condition, and full price are independently verified.
4. A human makes the final decision.

---

## 3. Out-of-Scope Uses

The model should not be used as an automated or authoritative system for:

- Certified vehicle appraisal
- Loan approval or credit decisions
- Insurance pricing or underwriting
- Tax or legal valuation
- Fully automated purchasing
- Fraud confirmation
- Mechanical diagnosis
- Vehicle safety assessment
- Guaranteed investment or resale decisions

The model was not designed or validated for high-stakes financial or regulatory decisions.

---

## 4. Target Definition

The prediction target is the vehicle listing's **advertised asking price**.

The model learns the relationship between structured listing characteristics and asking prices observed in historical Craigslist advertisements.

Conceptually:

    listing characteristics
            |
            v
    CatBoost regression model
            |
            v
    predicted market-consistent asking price

The target represents:

- The price shown in the advertisement
- Seller or dealer pricing behavior
- Historical marketplace listing patterns

The target does not directly represent:

- Final negotiated sale price
- Verified fair market value
- Dealer appraisal value
- Auction value
- Mechanical condition
- Total ownership cost
- Financing terms
- Down payment versus full price
- Guaranteed resale value

Because the source data contains asking prices rather than completed transactions, the system estimates how similar vehicles tend to be **listed**, not necessarily what buyers ultimately pay.

---

## 5. Dataset

The project uses the Craigslist used-cars and trucks dataset.

### Original Dataset

- Rows: 426,880
- Columns: 26

### Cleaned Dataset

- Rows: 361,956
- Columns: 22

The cleaning pipeline removed or handled records that violated the project's data-quality rules while preserving enough observations for modeling.

The dataset includes structured attributes such as:

- Listing identifier
- Price
- Year
- Manufacturer
- Model
- Condition
- Fuel type
- Odometer
- Title status
- Transmission
- Drive type
- Vehicle type
- Paint color
- State
- Posting date

Some fields contain missing, inconsistent, or incorrectly entered values.

---

## 6. Input Features

The final feature pipeline uses 21 predictive features derived from the cleaned listing data.

Feature groups include:

### Vehicle identity and category

- Manufacturer
- Model
- Vehicle type
- Fuel type
- Transmission
- Drive type

### Vehicle age and usage

- Model year
- Vehicle age
- Odometer-related information

### Vehicle condition and configuration

- Condition
- Title status
- Cylinders
- Paint color
- Other available structured attributes

### Geography and time

- State or regional information
- Listing-time features

The model does not use unnecessary personally identifying seller information.

---

## 7. Data Splitting Strategy

The project uses a **chronological split** rather than a random split.

### Main Splits

- Training set: 253,369 listings
- Calibration set: 43,600 listings
- Test set: 41,746 listings

Chronological splitting better reflects a realistic setting in which a model learns from earlier listings and is evaluated on later listings.

### VIN Leakage Prevention

VIN-based controls were used to reduce the chance that the same vehicle appeared in multiple splits.

Without this protection, a duplicate or reposted vehicle could appear in both training and testing, making performance look better than it really is.

Analogy:

> Testing a student with questions copied directly from the study guide does not measure genuine understanding. VIN leakage prevention helps keep the test set genuinely separate.

---

## 8. Baseline Models

Baseline models were created before selecting the final model.

A baseline establishes the minimum level of performance that a more complex model should exceed.

### 8.1 Segment Median Baseline

The segment median baseline predicted prices using median prices within relevant vehicle groups.

Test performance:

- Mean Absolute Error: $5,036.32
- Root Mean Squared Error: $9,204.85

This was a meaningful benchmark because it approximated a simple marketplace rule: look at the typical price of similar vehicles.

### 8.2 Ridge Regression

Ridge regression was evaluated as a regularized linear model.

Test performance:

- Mean Absolute Error: $5,164.40
- Root Mean Squared Error: $47,340.33

The Ridge model generated an extreme prediction of approximately $9.5 million for one listing, which severely increased RMSE.

This failure showed why the project uses:

- Multiple evaluation metrics
- Outlier inspection
- Baseline comparison
- Output validation
- Nonlinear modeling

---

## 9. Final Model

The selected model is **CatBoost regression**.

### Selected Hyperparameters

- Tree depth: 8
- Learning rate: 0.10
- Number of trees: 998

### Why CatBoost Was Selected

CatBoost is well suited to this dataset because it:

- Handles categorical features effectively
- Captures nonlinear relationships
- Learns interactions among variables
- Requires less manual encoding than many traditional models
- Performs strongly on structured tabular data

Examples of relationships it can learn include:

- Mileage affects different manufacturers differently
- Vehicle age interacts with condition
- Regional markets have different price levels
- Trucks, sedans, and SUVs depreciate differently
- The effect of model year depends on manufacturer and vehicle type

Analogy:

> A linear model tries to explain pricing with one mostly straight rule. CatBoost combines many decision trees, allowing the pricing rule to bend and adapt across different kinds of vehicles.

---

## 10. Model Performance

### Calibration Performance

- CatBoost calibration MAE: $3,240.38

### Test Performance

- Mean Absolute Error: $3,515.95
- Root Mean Squared Error: $7,351.12
- Median Absolute Error: $1,784.57

### Improvement Over Baseline

The CatBoost model improved test MAE by approximately **30.2%** compared with the segment median baseline.

---

## 11. Evaluation Metrics

### Mean Absolute Error

Mean Absolute Error measures the average absolute difference between observed and predicted asking prices.

    MAE = average(|actual price - predicted price|)

Final test MAE:

- $3,515.95

Interpretation:

> On average, the predicted asking price differed from the observed asking price by approximately $3,516.

### Root Mean Squared Error

Root Mean Squared Error gives larger penalties to large prediction errors.

    RMSE = square root(average((actual price - predicted price)^2))

Final test RMSE:

- $7,351.12

Because RMSE is considerably higher than MAE, some listings still produced large errors.

### Median Absolute Error

Median Absolute Error reports the middle absolute prediction error.

Final result:

- $1,784.57

Interpretation:

> Half of the test predictions were within approximately $1,785 of the observed asking price.

### Why Multiple Metrics Were Used

No single metric fully describes model behavior.

- MAE measures average dollar error
- RMSE highlights large failures
- Median Absolute Error describes the typical case
- Coverage measures uncertainty reliability

---

## 12. Uncertainty Estimation

The project uses **split conformal prediction** to produce uncertainty intervals around each point prediction.

A point prediction might say:

> The expected asking price is $25,000.

An uncertainty interval adds:

> Based on calibration errors, a model-consistent range may be approximately $18,000 to $32,000.

### Proper Training and Conformal Calibration

The original training portion was divided into:

- Proper training set: 202,695 listings
- Conformal calibration set: 50,674 listings

The model was trained on the proper training set. Its errors on the conformal calibration set were used to estimate uncertainty.

### Conformal Residual Quantile

For the nominal 90% interval:

- Calibration residual quantile: $6,981.67

A simplified interval is:

    lower bound = predicted price - conformal quantile
    upper bound = predicted price + conformal quantile

Additional project logic keeps interval bounds within valid price constraints.

### Analogy

Imagine an archer practicing before a competition.

- The point prediction is where the archer aims.
- Calibration errors show how far arrows usually land from the center.
- The conformal interval draws a range large enough to contain most previous misses.

Instead of pretending every prediction is exact, the model reports a calibrated range.

---

## 13. Uncertainty Metrics

### Target Coverage

- Nominal coverage: 90%

### Empirical Test Coverage

- Observed coverage: 87.88%

Coverage measures how often the observed asking price fell inside the predicted interval.

The observed coverage was below the 90% target, so the intervals were slightly overconfident on the later test period.

### Interval Width

- Average interval width: $13,204.86
- Median interval width: $13,963.33

A wider interval indicates greater uncertainty.

There is a tradeoff:

- Wider intervals usually improve coverage but provide less precision
- Narrower intervals provide more precision but may miss more often

---

## 14. Opportunity Classification

The project compares each listing's asking price with its conformal prediction interval.

### Potentially Underpriced

Rule:

    asking price < lower prediction bound

Count:

- 2,040 listings

Interpretation:

> The asking price is below the model's uncertainty-adjusted expected range.

### Market Consistent

Rule:

    lower prediction bound <= asking price <= upper prediction bound

Count:

- 36,686 listings

Interpretation:

> The asking price falls within the model-consistent range.

### Potentially Overpriced

Rule:

    asking price > upper prediction bound

Count:

- 3,020 listings

Interpretation:

> The asking price is above the model's uncertainty-adjusted expected range.

These three categories account for all 41,746 test listings.

The labels are model-relative signals, not verified statements about true vehicle value.

---

## 15. Signal Strength

Opportunity strength was based on how far the asking price fell outside the prediction interval.

Categories include:

- Weak signal
- Moderate signal
- Strong signal
- Market consistent

Conceptually:

    small gap outside interval -> weaker signal
    large gap outside interval -> stronger signal

A stronger signal means the listing is more unusual relative to the model. It does not prove that the listing is profitable, authentic, or mechanically sound.

---

## 16. Confidence Tiers

Confidence tiers were derived from interval width relative to the predicted price.

Conceptually:

    narrower relative interval -> higher confidence
    wider relative interval -> lower confidence

Confidence helps distinguish between:

- Predictions made in familiar and stable regions of the data
- Predictions with broad uncertainty
- Listings that require additional caution

High confidence does not guarantee that the listing information is accurate.

---

## 17. Tableau Decision-Support Layer

The test predictions were transformed into a Tableau-ready export containing:

- 41,746 scored listings
- 36 dashboard-ready fields

The dashboard includes:

- Listings analyzed
- Potential opportunities
- Market-consistent listings
- Potential overpricing signals
- Opportunity Explorer table
- Top manufacturers by opportunity count
- State filter
- Manufacturer filter
- Vehicle type filter
- Price band filter
- Confidence filter
- Signal strength filter

The packaged Tableau workbook is stored at:

    reports/tableau/marketplace_opportunity_dashboard.twbx

The dashboard helps users explore model output. It does not replace human review.

---

## 18. Limitations

### Asking Price Is Not Sale Price

The dataset records advertised asking prices rather than completed transaction prices.

Sellers may:

- Negotiate below the listed price
- Enter placeholder values
- Advertise a down payment rather than the full price
- Post an incorrect value
- Repost the same vehicle
- Never complete a sale

The model therefore learns listing behavior, not verified transaction value.

### Missing Mechanical Information

Structured fields cannot fully describe:

- Engine condition
- Transmission condition
- Accident damage
- Rust
- Tire condition
- Interior condition
- Maintenance history
- Inspection results
- Hidden defects
- Aftermarket modifications

Two vehicles with similar structured attributes may have very different real-world values.

### No Image Understanding

The model does not inspect listing photographs.

It cannot detect:

- Visible body damage
- Missing parts
- Poor interior condition
- Flood damage clues
- Misleading photographs
- Images copied from another listing

### Limited Listing-Text Understanding

Important information may appear only in free-text descriptions, including:

- "Needs transmission"
- "Parts only"
- "Does not run"
- "Salvage title"
- "Down payment"
- "Monthly payment"
- "Call for actual price"

The structured model may flag these listings because it cannot fully understand those explanations.

### Data Quality Problems

Marketplace data may contain:

- Duplicate listings
- Missing values
- Incorrect years
- Incorrect odometer values
- Placeholder prices
- Financing advertisements
- Dealer spam
- Typographical errors
- Reposted vehicles
- Inconsistent category labels

Cleaning and validation reduce these problems but cannot eliminate all of them.

### Geographic Differences

Vehicle markets vary by:

- State
- City
- Local supply
- Local income
- Fuel prices
- Weather
- Regional preferences
- Seasonal demand

The model may perform worse in regions with limited or unusual training data.

### Temporal Drift

Vehicle prices change over time because of:

- Inflation
- Interest rates
- Fuel prices
- Supply-chain disruptions
- New-vehicle availability
- Seasonal demand
- Economic conditions

The model should be retrained before being applied to substantially newer market data.

### Rare Categories

The model may be less reliable for:

- Rare manufacturers
- Exotic vehicles
- Classic vehicles
- Collectible vehicles
- Limited editions
- Heavily modified vehicles
- Very old vehicles
- Very new vehicles
- Extreme mileage values

Machine-learning models usually perform best on examples similar to their training data.

### Uneven Conditional Coverage

Overall empirical coverage was 87.88%, but coverage was not uniform across all subgroups.

The $50,000-and-above price segment showed severe undercoverage.

Extra caution is required for:

- Luxury vehicles
- Rare vehicles
- High-performance vehicles
- Collector vehicles
- Very high-priced listings

A global conformal interval does not guarantee equal coverage for every subgroup.

---

## 19. Known Failure Cases

### Placeholder Prices

Listings priced at values such as $0, $1, $100, or $999 may be flagged as extreme opportunities even when the displayed price is not real.

### Financing and Down-Payment Advertisements

A dealer may advertise a value such as:

    $1,500 down

rather than the full purchase price.

The model may interpret this as an unusually low asking price.

### Damaged or Non-Running Vehicles

A vehicle may be correctly priced far below normal market levels because it:

- Does not run
- Needs an engine
- Needs a transmission
- Has major accident damage
- Is sold for parts
- Has a salvage or rebuilt title

The structured fields may not capture these facts.

### Incorrect Odometer Values

Possible problems include:

- One mile entered for a heavily used vehicle
- Several million miles
- Kilometers entered as miles
- Missing digits
- Default values

Incorrect mileage can distort predictions.

### Incorrect Year or Model

A seller may choose the wrong:

- Model year
- Manufacturer
- Model
- Vehicle type
- Transmission
- Fuel type

The model assumes that input fields are reasonably accurate.

### Rare and Collectible Vehicles

Ordinary depreciation patterns may not apply to:

- Classic cars
- Collector vehicles
- Limited editions
- Restored vehicles
- Modified vehicles
- Exotic vehicles

These vehicles may be priced based on rarity, originality, or specialized condition.

### High-Priced Vehicles

The 90% intervals showed severe undercoverage in the $50,000-and-above segment.

Opportunity signals for expensive vehicles require additional scrutiny.

### Extreme Linear-Model Predictions

The Ridge experiment produced an approximately $9.5 million prediction for one listing.

Ridge was not selected as the final model, but this failure demonstrated why the project includes multiple metrics, outlier inspection, nonlinear modeling, and validation.

---

## 20. Ethical Cautions

### Do Not Present Predictions as Objective Truth

The model estimates patterns found in historical marketplace listings.

Historical prices may reflect:

- Regional economic inequality
- Dealer pricing strategies
- Unequal access to transportation
- Differences in market power
- Local supply constraints

The prediction should not be described as an objectively fair or universally correct price.

### Avoid High-Stakes Financial Use

The model should not be used for:

- Loan approval
- Credit decisions
- Insurance pricing
- Underwriting
- Repossession decisions
- Regulatory appraisal
- Tax assessment

These uses would require substantially stronger validation, governance, legal review, and monitoring.

### Avoid Fully Automated Purchasing

A potentially underpriced signal should trigger investigation, not an automatic purchase.

Users should independently verify:

- VIN
- Title status
- Seller identity
- Listing authenticity
- Mechanical condition
- Accident history
- Maintenance records
- Full asking price
- Financing terms
- Ownership documentation

### Scam and Fraud Risk

An extremely low price may indicate:

- A fake listing
- Stolen photographs
- A deposit scam
- A fake shipping arrangement
- Identity theft
- A misleading financing advertisement
- A nonexistent vehicle

The model does not detect fraud.

A strong underpricing signal may increase the need for caution rather than reduce it.

### Privacy

Public outputs should exclude unnecessary personal information, including:

- Phone numbers
- Email addresses
- Seller names
- Exact personal addresses
- Other personally identifying information

Only fields necessary for analysis should be published.

### Human Oversight

Every flagged listing should be reviewed by a person before action is taken.

The intended process is:

    model signal
         |
         v
    manual investigation
         |
         v
    independent verification
         |
         v
    human decision

### Responsible Interpretation

Correct interpretation:

> The asking price is unusual relative to the model's learned marketplace patterns and estimated uncertainty.

Incorrect interpretation:

> The model proved that the listing is a bargain or that the seller is overcharging.

---

## 21. Reproducibility and Testing

The repository includes:

- Data ingestion
- Data validation
- Data cleaning
- Chronological splitting
- VIN leakage controls
- Feature engineering
- Baseline modeling
- Ridge regression evaluation
- CatBoost training
- Conformal calibration
- Opportunity classification
- Tableau export generation
- Automated tests

At the completion of Phase 7:

- 75 tests were passing

Automated tests help verify:

- Required columns
- Valid row counts
- Unique identifiers
- Correct opportunity categories
- Absence of invalid numeric values
- Stable export behavior

---

## 22. Monitoring Recommendations

If the model were deployed or refreshed, monitoring should include:

- MAE over time
- RMSE over time
- Median Absolute Error
- Overall interval coverage
- Coverage by price band
- Coverage by manufacturer
- Coverage by state
- Coverage by vehicle type
- Interval width
- Missing-value rates
- Category-frequency shifts
- Odometer distribution shifts
- Price distribution shifts
- Rate of extreme opportunity signals

A major drop in coverage or increase in error should trigger investigation and possible retraining.

---

## 23. Recommended Future Improvements

### Data Improvements

- Add completed sale-price data
- Add vehicle-history information
- Add accident and title-history data
- Add regional economic features
- Add newer marketplace data
- Improve duplicate and repost detection

### Text and Image Features

- Analyze listing descriptions using natural language processing
- Detect financing and down-payment language
- Detect damaged or non-running descriptions
- Extract condition information from photographs
- Detect reused or suspicious images

### Modeling Improvements

- Compare CatBoost with LightGBM and XGBoost
- Evaluate specialized models by price band
- Build separate models for rare or luxury vehicles
- Tune models using time-aware cross-validation
- Evaluate quantile regression methods

### Uncertainty Improvements

- Use group-conditional conformal calibration
- Use price-band-specific calibration
- Use manufacturer-specific calibration where sufficient data exists
- Improve coverage for vehicles above $50,000
- Investigate adaptive conformal methods

### Product Improvements

- Publish the dashboard to Tableau Public
- Add listing links when safely available
- Add explanation panels for each signal
- Add subgroup reliability warnings
- Add data-freshness indicators

---

## 24. Final Summary

This system predicts market-consistent asking prices for used vehicle listings and adds uncertainty-aware opportunity signals.

Its strongest use is prioritization:

- Which listings deserve closer investigation?
- Which asking prices appear unusual?
- Which predictions have wider or narrower uncertainty?
- Which marketplace segments contain more flagged listings?

The project should not be treated as a certified valuation tool, fraud detector, professional appraisal, or substitute for vehicle inspection and human judgment.

The central principle is:

> Use the model to decide what to investigate, not what to believe automatically.