# Real Estate Investment Advisor

End-to-end project: classify whether a listing is a **Good Investment** (0/1) and estimate **price after 5 years**, with EDA, MLflow tracking, and a Streamlit app.

**Limitation:** `india_housing_prices.csv` looks synthetic. Prices are similar across cities. Do not treat charts or predictions as real Indian market research. See `docs/findings.md`.

## Setup

```bash
cd real_estate_advisor
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run every command below from `real_estate_advisor/` so `python -m src...` can import the package.

## Phase 1–2 — Clean, features, targets

```bash
python -m src.preprocessing
```

Writes `data/processed/cleaned_housing.csv` and `reports/preprocessing_report.md`.

- Recomputes ₹/sqft from lakhs and size (original column is discarded for analysis).
- Flags invalid floors (~46.5%), caps floor for `floor_ratio`, does **not** drop rows.
- Builds amenity flags, infrastructure proxy, city medians, good-investment label, 5-year price.

## Phase 3 — EDA (20 questions)

```bash
python -m src.eda
```

Charts: `reports/figures/`. Insights: `reports/eda_insights.md`.

## Phase 4 — Train + MLflow

```bash
python -m src.train_classifier
python -m src.train_regressor
```

- Pipelines: `models/good_investment_pipeline.joblib`, `models/future_price_pipeline.joblib`
- Comparison table: `reports/model_comparison.md`
- Tracking UI (file store is allowed via `MLFLOW_ALLOW_FILE_STORE=true` in the train scripts):

```bash
set MLFLOW_ALLOW_FILE_STORE=true
mlflow ui --backend-store-uri ./mlruns
```

Experiments: `good_investment_classification`, `future_price_regression`.  
Registered names: `GoodInvestmentClassifier`, `FuturePriceRegressor`.

On 250k rows the train scripts take several minutes (tree models + a 30k-row randomized search).

## Phase 5 — App

```bash
streamlit run app/streamlit_app.py
```

Sidebar pages: **Predict**, **Explore**, **Insights**.

Generate the PDF project report from the saved findings and figures:

```bash
python reports/generate_project_report.py
```

Output: `reports/real_estate_investment_report.pdf`.

## Deploy to Streamlit Community Cloud

Push this project directory to GitHub, then create an app at
[share.streamlit.io](https://share.streamlit.io) with:

- Main file path: `app/streamlit_app.py`
- Python dependencies: `requirements.txt`

Keep `data/processed/cleaned_housing.csv` in the repository; the app needs it
for all pages. Model binaries and MLflow history are local-only. Predict uses
the documented rule-based assumptions until trained model pipelines are added.

## Layout

```
real_estate_advisor/
  data/raw/  data/processed/
  src/preprocessing.py  features.py  eda.py
      train_classifier.py  train_regressor.py  evaluate.py  utils.py
  app/streamlit_app.py
  models/
  reports/figures/
  docs/methodology.md  findings.md  use_cases.md
```
