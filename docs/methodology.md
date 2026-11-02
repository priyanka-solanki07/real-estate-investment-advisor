# Methodology

## Data

Source file: `data/raw/india_housing_prices.csv` (250,000 rows, 23 columns, 0 missing values, 0 duplicates).

Columns used as listed in the project brief. There is **no crime rate**, **no official infrastructure index**, and **no listing date**. Those fields are not invented.

## Cleaning

1. **Price per sqft.** The original `Price_per_SqFt` is in lakhs per square foot and rounded to 2 decimals, so it is unreliable. Analysis and models use  
   `Price_per_SqFt_INR = (Price_in_Lakhs * 100000) / Size_in_SqFt`.
2. **Floors.** `Floor_No > Total_Floors` on about 46.5% of rows. Rows are **not dropped**. We add `floor_invalid` and cap `Floor_No` at `Total_Floors` only to compute `floor_ratio`. Dropping half the sample would dominate every later result; the two fields look independently generated.
3. **Age.** `Age_of_Property` equals `2025 - Year_Built` on this file. We recompute age with reference year 2025 and store `age_mismatch`.
4. **Amenities.** The comma-separated string is split into `amenity_count` and one-hot flags (`Gym`, `Pool`, `Garden`, `Playground`, `Clubhouse`).
5. **Outliers.** IQR flags on size, list price, and recomputed ₹/sqft. Flags are stored; rows are not removed. Size and list price sit inside the documented bounds, so IQR finds no outliers there. ₹/sqft has a right tail (small homes with high prices).
6. **Infrastructure proxy.** `infrastructure_score` = min-max of (`Nearby_Schools` + `Nearby_Hospitals` + transport mapped Low/Medium/High → 1/2/3). Range 3–23 mapped to 0–1.

## Targets

**Good Investment (classification).** Six binary points, label = 1 if score ≥ `GOOD_INVESTMENT_THRESHOLD` (default 3):

| Point | Rule |
|---|---|
| 1 | ₹/sqft ≤ city median ₹/sqft |
| 2 | BHK ≥ 3 |
| 3 | Ready_to_Move |
| 4 | infrastructure_score > median |
| 5 | amenity_count > median |
| 6 | Public transport = High |

**Future price (regression).** `Future_Price_5Y_Lakhs = Price_in_Lakhs * (1 + r)^5`.

`r` is **not** a single national rate. It is assembled from `src/utils.py` → `APPRECIATION_CONFIG`:

- City tier base: tier 1 = 10%, tier 2 = 8%, tier 3 = 6%. Tiers follow a conventional metro grouping, **not** prices in this file (those prices barely differ by city).
- Property type add-on: Apartment +0, Independent House +0.8 pp, Villa +1.5 pp.
- Infrastructure add-on: `infrastructure_score * 2 pp`.
- Clip to [5%, 13%].

These rates are modelling assumptions for a synthetic table, not a market forecast.

## Leakage

The classification label is a **deterministic rule** of six columns. To avoid a trivial copy of the rule, the classifier pipeline **excludes**:

- `investment_score`, `Good_Investment`
- `is_ready_to_move`, `amenity_count`, `infrastructure_score`, `city_median_pps`
- the other task’s `Future_Price_5Y_Lakhs` and `appreciation_rate`
- `city_tier`

Raw ingredients that remain (BHK, availability, schools, hospitals, transport, price, size) can still reconstruct much of the score. **High classification metrics are therefore expected and should not be sold as “the model discovered investment quality.”**

The regression target is a **known multiplicative transform** of current price using a rate table driven by city, type, and infrastructure. The regressor **excludes** `appreciation_rate`, `city_tier`, and the classification targets, but it **keeps** current price (required at inference). **Very high R2 is expected.** Honest use of the app: “illustrative 5-year value under the documented rate table,” not a market appraisal.

Locality (~500 values) is **target-encoded inside the sklearn pipeline** with CV (`TargetEncoder`), so encodings are fit on training folds only. City median price/₹/sqft used as model features are fit on the training fold via `HousingFeatureEngineer`.

## Models

- Shared `Pipeline`: `HousingFeatureEngineer` → `ColumnTransformer` (scale numerics, ordinal encode transport/furnishing, one-hot low-cardinality cats, target-encode State/City/Locality) → estimator.
- Split: 80/20, `random_state=42`, stratified for classification.
- Classification: Logistic Regression, Random Forest, XGBoost. Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
- Regression: Linear Regression, Random Forest, XGBoost. Metrics: RMSE, MAE, R2.
- Light `RandomizedSearchCV` (6 draws, 3-fold, 30k-row sample) on the tree models.
- MLflow experiments: `good_investment_classification`, `future_price_regression`. Best of each registered as `GoodInvestmentClassifier` and `FuturePriceRegressor`.
- Final pipelines saved with joblib under `models/`.
