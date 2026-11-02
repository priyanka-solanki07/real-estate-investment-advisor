"""Load, validate, clean, engineer, attach targets, and save the processed dataset."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.features import add_targets, engineer_frame
from src.utils import (
    GOOD_INVESTMENT_THRESHOLD,
    data_processed_path,
    data_raw_path,
    iqr_outlier_mask,
    reports_dir,
)


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or data_raw_path()
    if not path.exists():
        fallback = ROOT.parent / "india_housing_prices.csv"
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(f"Raw CSV not found at {path}")
    return pd.read_csv(path)


def validate(df: pd.DataFrame) -> dict:
    expected = {
        "ID",
        "State",
        "City",
        "Locality",
        "Property_Type",
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Year_Built",
        "Furnished_Status",
        "Floor_No",
        "Total_Floors",
        "Age_of_Property",
        "Nearby_Schools",
        "Nearby_Hospitals",
        "Public_Transport_Accessibility",
        "Parking_Space",
        "Security",
        "Amenities",
        "Facing",
        "Owner_Type",
        "Availability_Status",
    }
    missing_cols = sorted(expected - set(df.columns))
    extra_cols = sorted(set(df.columns) - expected)
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "n_cities": int(df["City"].nunique()),
        "n_states": int(df["State"].nunique()),
        "n_localities": int(df["Locality"].nunique()),
    }


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """IQR flags only — outliers are reported, not dropped."""
    out = df.copy()
    out["outlier_size"] = iqr_outlier_mask(out["Size_in_SqFt"]).astype(int)
    out["outlier_price"] = iqr_outlier_mask(out["Price_in_Lakhs"]).astype(int)
    out["outlier_pps"] = iqr_outlier_mask(out["Price_per_SqFt_INR"]).astype(int)
    return out


def build_cleaned(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    info = validate(df)
    floor_invalid_pct = float((df["Floor_No"] > df["Total_Floors"]).mean() * 100)

    engineered, _ = engineer_frame(df)
    engineered = flag_outliers(engineered)
    engineered = add_targets(engineered)

    outlier_summary = {
        "size": int(engineered["outlier_size"].sum()),
        "price": int(engineered["outlier_price"].sum()),
        "pps": int(engineered["outlier_pps"].sum()),
        "pps_pct": float(engineered["outlier_pps"].mean() * 100),
    }
    class_balance = engineered["Good_Investment"].value_counts(normalize=True).to_dict()
    class_counts = engineered["Good_Investment"].value_counts().to_dict()
    age_mismatch_pct = float(engineered["age_mismatch"].mean() * 100)

    info.update(
        {
            "floor_invalid_pct": floor_invalid_pct,
            "outliers": outlier_summary,
            "good_investment_threshold": GOOD_INVESTMENT_THRESHOLD,
            "class_counts": {int(k): int(v) for k, v in class_counts.items()},
            "class_balance": {int(k): float(v) for k, v in class_balance.items()},
            "age_mismatch_pct": age_mismatch_pct,
            "mean_appreciation_rate": float(engineered["appreciation_rate"].mean()),
        }
    )
    return engineered, info


def write_preprocessing_report(info: dict) -> Path:
    path = reports_dir() / "preprocessing_report.md"
    reports_dir().mkdir(parents=True, exist_ok=True)
    balance = info["class_balance"]
    counts = info["class_counts"]
    lines = [
        "# Preprocessing report",
        "",
        f"- Rows: {info['n_rows']:,} | Columns (raw): {info['n_cols']}",
        f"- Missing values: {info['missing_values']} | Duplicate rows: {info['duplicate_rows']}",
        f"- States: {info['n_states']} | Cities: {info['n_cities']} | Localities: {info['n_localities']}",
        "",
        "## Price per sqft",
        "The source `Price_per_SqFt` is in **lakhs per sqft** and rounded to 2 decimals, so it is discarded for analysis.",
        "We recompute `Price_per_SqFt_INR` = `(Price_in_Lakhs * 100000) / Size_in_SqFt` and use that everywhere.",
        "",
        "## Floor inconsistency",
        f"- `Floor_No > Total_Floors` on **{info['floor_invalid_pct']:.2f}%** of rows.",
        "- Decision: **keep all rows**, add `floor_invalid`, and **cap** `Floor_No` at `Total_Floors` for `floor_ratio`.",
        "- Why: dropping ~46% of the data would bias every later step. The pattern looks like independently generated fields, not real stacked buildings. The flag lets a model use the inconsistency if it has signal; the cap keeps `floor_ratio` in [0, 1].",
        "",
        "## Age",
        f"- `Age_of_Property` matches `{2025} - Year_Built` on all but {info['age_mismatch_pct']:.4f}% of rows (reference year 2025).",
        "- We still store `Age_recomputed` and `age_mismatch`.",
        "",
        "## Outliers (IQR, not removed)",
        f"- Size_in_SqFt: {info['outliers']['size']:,}",
        f"- Price_in_Lakhs: {info['outliers']['price']:,}",
        f"- Price_per_SqFt_INR: {info['outliers']['pps']:,} ({info['outliers']['pps_pct']:.2f}%)",
        "Size and list price sit inside the documented bounds, so IQR finds no outliers there. High ₹/sqft rows are small homes with high prices; they are flagged (`outlier_pps`) and kept.",
        "",
        "## Targets",
        f"- Good Investment threshold: score >= {info['good_investment_threshold']}",
        f"- Class counts: {counts}",
        f"- Class balance (share): 0 → {balance.get(0, 0):.3f}, 1 → {balance.get(1, 0):.3f}",
        f"- Mean 5-year annual appreciation rate used for the regression target: {info['mean_appreciation_rate']:.4f}",
        "",
        "## Synthetic-data limitation",
        "Prices are very similar across cities. Location 'effects' in this file should not be read as real Indian housing-market structure.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run() -> pd.DataFrame:
    raw = load_raw()
    cleaned, info = build_cleaned(raw)
    out_path = data_processed_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)
    report_path = write_preprocessing_report(info)
    print("=== Validation ===")
    print(f"rows={info['n_rows']:,} cols={info['n_cols']} missing={info['missing_values']} dups={info['duplicate_rows']}")
    print(f"states={info['n_states']} cities={info['n_cities']} localities={info['n_localities']}")
    print(f"floor_invalid={info['floor_invalid_pct']:.2f}% (flagged + capped, not dropped)")
    print(f"IQR outliers: size={info['outliers']['size']} price={info['outliers']['price']} pps={info['outliers']['pps']} ({info['outliers']['pps_pct']:.2f}%)")
    print(f"age_mismatch={info['age_mismatch_pct']:.4f}%")
    print("=== Good Investment class balance ===")
    print(f"counts={info['class_counts']} shares={ {k: round(v, 4) for k, v in info['class_balance'].items()} }")
    print(f"mean appreciation rate={info['mean_appreciation_rate']:.4f}")
    print(f"saved {out_path}")
    print(f"report {report_path}")
    return cleaned


if __name__ == "__main__":
    run()
