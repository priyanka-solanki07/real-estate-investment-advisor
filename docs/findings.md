# Findings

Full question-by-question writeup: `reports/eda_insights.md`. Preprocessing numbers: `reports/preprocessing_report.md`. Model table: `reports/model_comparison.md`.

## Data quality

- Original `Price_per_SqFt` is lakhs/sqft and too coarse; recomputed ₹/sqft has mean around ₹13,000 with a long right tail (~8% IQR outliers). Those rows were kept.
- Floor numbers are invalid on ~46.5% of listings. Flagged and capped for ratio; nothing dropped.
- Age matches 2025 − Year_Built.

## What the table does *not* show

List prices and ₹/sqft are **almost the same across 42 cities and 20 states**. Amenities, parking, facing, furnishing, schools, and hospitals do not move price. That is a synthetic-data artefact. Do not quote these charts as Indian market research.

The only sharp “relationship” that is guaranteed by construction is the good-investment rate vs High public transport / BHK / ready-to-move, because those fields *are* the label.

## Modelling takeaway

Treat the Streamlit output as a demo of an end-to-end pipeline (clean → engineer → track → serve), not as investment advice. The 5-year price follows the documented rate table. The good-investment flag follows the documented 6-point score.
