# Preprocessing report

- Rows: 250,000 | Columns (raw): 23
- Missing values: 0 | Duplicate rows: 0
- States: 20 | Cities: 42 | Localities: 500

## Price per sqft
The source `Price_per_SqFt` is in **lakhs per sqft** and rounded to 2 decimals, so it is discarded for analysis.
We recompute `Price_per_SqFt_INR` = `(Price_in_Lakhs * 100000) / Size_in_SqFt` and use that everywhere.

## Floor inconsistency
- `Floor_No > Total_Floors` on **46.52%** of rows.
- Decision: **keep all rows**, add `floor_invalid`, and **cap** `Floor_No` at `Total_Floors` for `floor_ratio`.
- Why: dropping ~46% of the data would bias every later step. The pattern looks like independently generated fields, not real stacked buildings. The flag lets a model use the inconsistency if it has signal; the cap keeps `floor_ratio` in [0, 1].

## Age
- `Age_of_Property` matches `2025 - Year_Built` on all but 0.0000% of rows (reference year 2025).
- We still store `Age_recomputed` and `age_mismatch`.

## Outliers (IQR, not removed)
- Size_in_SqFt: 0
- Price_in_Lakhs: 0
- Price_per_SqFt_INR: 19,723 (7.89%)
Size and list price sit inside the documented bounds, so IQR finds no outliers there. High ₹/sqft rows are small homes with high prices; they are flagged (`outlier_pps`) and kept.

## Targets
- Good Investment threshold: score >= 3
- Class counts: {1: 146530, 0: 103470}
- Class balance (share): 0 → 0.414, 1 → 0.586
- Mean 5-year annual appreciation rate used for the regression target: 0.0978

## Synthetic-data limitation
Prices are very similar across cities. Location 'effects' in this file should not be read as real Indian housing-market structure.
