# Initial Data Dictionary

The final dictionary will be regenerated after inspecting the downloaded CSV. The expected Craigslist schema is listed below.

| Field | Expected type | Role | Initial treatment |
|---|---|---|---|
| `id` | integer/string | Identifier | Retain as listing ID; exclude from model |
| `url` | string | Identifier | Exclude from model; may support duplicate audit |
| `region` | string | Categorical feature | Normalize text |
| `region_url` | string | Identifier/geography | Usually exclude from model |
| `price` | numeric | Target | Validate and clean; never use as feature |
| `year` | numeric | Feature | Validate; derive vehicle age |
| `manufacturer` | string | Categorical feature | Normalize missing values |
| `model` | string | High-cardinality categorical feature | Normalize carefully |
| `condition` | string | Categorical feature | Preserve missingness |
| `cylinders` | string | Categorical feature | Parse or retain categorically |
| `fuel` | string | Categorical feature | Normalize categories |
| `odometer` | numeric | Numeric feature | Validate; derive mileage per year |
| `title_status` | string | Categorical feature | Retain; important risk indicator |
| `transmission` | string | Categorical feature | Normalize categories |
| `VIN` | string | Identifier | Use for duplicate audit; exclude from model |
| `drive` | string | Categorical feature | Normalize categories |
| `size` | string | Categorical feature | Preserve missingness |
| `type` | string | Categorical feature | Normalize categories |
| `paint_color` | string | Categorical feature | Preserve missingness |
| `image_url` | string | Identifier | Exclude from model |
| `description` | string | Free text | Exclude from first model; audit for leakage |
| `county` | string/numeric | Geography | Inspect missingness before use |
| `state` | string | Categorical/geographic feature | Normalize uppercase code |
| `lat` | numeric | Geographic feature | Validate plausible coordinates |
| `long` | numeric | Geographic feature | Validate plausible coordinates |
| `posting_date` | datetime | Time feature/split key | Parse to UTC-aware datetime |

## Derived fields planned

| Field | Definition |
|---|---|
| `vehicle_age` | Listing year minus model year |
| `mileage_per_year` | Odometer divided by max(vehicle age, 1) |
| `listing_month` | Calendar month from posting date |
| `listing_day_of_week` | Day of week from posting date |
| `listing_completeness` | Count or percentage of selected populated fields |
| `model_clean` | Normalized model text |
| `location_valid` | Whether latitude and longitude are plausible |

## Fields requiring verification after download

- Exact column names and types
- Whether `posting_date` is consistently populated
- Whether the dataset version contains 26 columns
- Dataset date range
- Duplicate rate by `id`, `VIN`, URL, and vehicle attributes
- Missingness and validity thresholds
