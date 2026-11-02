"""Feature engineering, target construction, and sklearn pipelines."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from src.utils import (
    AMENITY_VOCAB,
    APPRECIATION_CONFIG,
    CLASSIFIER_EXCLUDE,
    FURNISHED_ORDINAL,
    GOOD_INVESTMENT_THRESHOLD,
    RAW_INPUT_COLUMNS,
    REFERENCE_YEAR,
    REGRESSOR_EXCLUDE,
    TRANSPORT_ORDINAL,
    city_tier,
    parse_amenities,
    yes_no_to_int,
)


def recompute_price_per_sqft_inr(df: pd.DataFrame) -> pd.Series:
    """Rupees per sqft: (Price_in_Lakhs * 100000) / Size_in_SqFt."""
    size = df["Size_in_SqFt"].replace(0, np.nan)
    return (df["Price_in_Lakhs"] * 100_000) / size


def add_amenity_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["Amenities"].map(parse_amenities)
    out["amenity_count"] = parsed.map(len)
    for name in AMENITY_VOCAB:
        out[f"amenity_{name}"] = parsed.map(lambda items, n=name: int(n in items))
    return out


def add_floor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flag invalid floors; cap Floor_No for ratio only.

    Why cap instead of drop: Floor_No > Total_Floors on ~46% of rows. Dropping
    them would throw away almost half the sample. The two columns look independently
    generated (synthetic). We keep every row, set floor_invalid=1, and cap Floor_No
    at Total_Floors so floor_ratio stays in [0, 1] and is usable as a feature.
    """
    out = df.copy()
    total = out["Total_Floors"].clip(lower=1)
    out["floor_invalid"] = (out["Floor_No"] > out["Total_Floors"]).astype(int)
    out["Floor_No_capped"] = np.minimum(out["Floor_No"], total)
    out["floor_ratio"] = out["Floor_No_capped"] / total
    return out


def add_infrastructure_score(df: pd.DataFrame) -> pd.DataFrame:
    """Proxy only: schools + hospitals + transport (1/2/3), then min-max to 0-1.

    There is no infrastructure index in the source data; this is an explicit proxy.
    """
    out = df.copy()
    transport = out["Public_Transport_Accessibility"].map(TRANSPORT_ORDINAL).fillna(2)
    raw = out["Nearby_Schools"] + out["Nearby_Hospitals"] + transport
    # theoretical min=1+1+1=3, max=10+10+3=23
    out["infrastructure_score"] = (raw - 3) / (23 - 3)
    out["transport_ordinal"] = transport.astype(int)
    return out


def add_age_features(df: pd.DataFrame, reference_year: int = REFERENCE_YEAR) -> pd.DataFrame:
    out = df.copy()
    out["Age_recomputed"] = reference_year - out["Year_Built"]
    if "Age_of_Property" in out.columns:
        out["age_mismatch"] = (out["Age_recomputed"] != out["Age_of_Property"]).astype(int)
    else:
        out["age_mismatch"] = 0
    return out


def add_city_stats(df: pd.DataFrame, stats: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    pps = out["Price_per_SqFt_INR"] if "Price_per_SqFt_INR" in out.columns else recompute_price_per_sqft_inr(out)
    out["Price_per_SqFt_INR"] = pps
    if stats is None:
        stats = (
            out.groupby("City", as_index=True)
            .agg(
                city_median_price=("Price_in_Lakhs", "median"),
                city_median_pps=("Price_per_SqFt_INR", "median"),
            )
        )
    out = out.merge(stats, left_on="City", right_index=True, how="left")
    out["city_median_price"] = out["city_median_price"].fillna(out["Price_in_Lakhs"].median())
    out["city_median_pps"] = out["city_median_pps"].fillna(out["Price_per_SqFt_INR"].median())
    return out, stats


def add_simple_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_ready_to_move"] = (out["Availability_Status"] == "Ready_to_Move").astype(int)
    out["bhk_per_1000sqft"] = out["BHK"] / (out["Size_in_SqFt"] / 1000.0)
    out["furnished_ordinal"] = out["Furnished_Status"].map(FURNISHED_ORDINAL).fillna(1).astype(int)
    out["parking_bin"] = yes_no_to_int(out["Parking_Space"])
    out["security_bin"] = yes_no_to_int(out["Security"])
    out["city_tier"] = out["City"].map(city_tier)
    return out


def engineer_frame(df: pd.DataFrame, city_stats: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full-data engineering used for the cleaned CSV / EDA (not the CV pipeline)."""
    out = df.copy()
    if "Price_per_SqFt" in out.columns:
        out["Price_per_SqFt_original"] = out["Price_per_SqFt"]
    out["Price_per_SqFt_INR"] = recompute_price_per_sqft_inr(out)
    out = add_floor_features(out)
    out = add_amenity_features(out)
    out = add_infrastructure_score(out)
    out = add_age_features(out)
    out = add_simple_flags(out)
    out, stats = add_city_stats(out, stats=city_stats)
    return out, stats


def add_targets(df: pd.DataFrame, threshold: int = GOOD_INVESTMENT_THRESHOLD) -> pd.DataFrame:
    """Build both modelling targets.

    Classification score (1 point each, label=1 if score >= threshold):
      1. price_per_sqft <= city median pps
      2. BHK >= 3
      3. Ready_to_Move
      4. infrastructure_score above median
      5. amenity_count above median
      6. High public transport

    Regression: Future_Price_5Y_Lakhs = Price_in_Lakhs * (1 + r) ** 5
    where r = clip(tier_base + property_type_adj + infra_score * infra_adj_max).
    """
    out = df.copy()
    infra_median = out["infrastructure_score"].median()
    amenity_median = out["amenity_count"].median()

    p1 = (out["Price_per_SqFt_INR"] <= out["city_median_pps"]).astype(int)
    p2 = (out["BHK"] >= 3).astype(int)
    p3 = out["is_ready_to_move"]
    p4 = (out["infrastructure_score"] > infra_median).astype(int)
    p5 = (out["amenity_count"] > amenity_median).astype(int)
    p6 = (out["Public_Transport_Accessibility"] == "High").astype(int)

    out["investment_score"] = p1 + p2 + p3 + p4 + p5 + p6
    out["Good_Investment"] = (out["investment_score"] >= threshold).astype(int)

    cfg = APPRECIATION_CONFIG
    tier_rate = out["city_tier"].map(cfg["tier_base"]).astype(float)
    type_adj = out["Property_Type"].map(cfg["property_type_adj"]).fillna(0.0)
    infra_adj = out["infrastructure_score"].clip(0, 1) * cfg["infra_adj_max"]
    rate = (tier_rate + type_adj + infra_adj).clip(cfg["min_rate"], cfg["max_rate"])
    out["appreciation_rate"] = rate
    years = cfg["horizon_years"]
    out["Future_Price_5Y_Lakhs"] = out["Price_in_Lakhs"] * ((1 + rate) ** years)
    return out


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible engineer so training and Streamlit share one pipeline.

    City medians are learned on the training fold only (no test leakage).
    """

    def __init__(self, reference_year: int = REFERENCE_YEAR):
        self.reference_year = reference_year
        self.city_stats_: Optional[pd.DataFrame] = None
        self.global_median_price_: Optional[float] = None
        self.global_median_pps_: Optional[float] = None

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).copy()
        frame["Price_per_SqFt_INR"] = recompute_price_per_sqft_inr(frame)
        self.global_median_price_ = float(frame["Price_in_Lakhs"].median())
        self.global_median_pps_ = float(frame["Price_per_SqFt_INR"].median())
        _, self.city_stats_ = add_city_stats(frame)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        for col in RAW_INPUT_COLUMNS:
            if col not in frame.columns:
                frame[col] = np.nan if col not in {"Amenities"} else ""
        frame["Price_per_SqFt_INR"] = recompute_price_per_sqft_inr(frame)
        frame = add_floor_features(frame)
        frame = add_amenity_features(frame)
        frame = add_infrastructure_score(frame)
        frame = add_age_features(frame, reference_year=self.reference_year)
        frame = add_simple_flags(frame)
        frame, _ = add_city_stats(frame, stats=self.city_stats_)
        frame["city_median_price"] = frame["city_median_price"].fillna(self.global_median_price_)
        frame["city_median_pps"] = frame["city_median_pps"].fillna(self.global_median_pps_)
        return frame


def _feature_columns_for_task(task: str) -> dict[str, list[str]]:
    numeric = [
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Price_per_SqFt_INR",
        "Year_Built",
        "Age_recomputed",
        "Total_Floors",
        "Floor_No_capped",
        "floor_ratio",
        "floor_invalid",
        "Nearby_Schools",
        "Nearby_Hospitals",
        "bhk_per_1000sqft",
        "city_median_price",
        "parking_bin",
        "security_bin",
        "age_mismatch",
        "furnished_ordinal",
        "transport_ordinal",
    ]
    amenity_flags = [f"amenity_{name}" for name in AMENITY_VOCAB]
    numeric.extend(amenity_flags)

    if task == "regressor":
        numeric.extend(
            [
                "infrastructure_score",
                "amenity_count",
                "is_ready_to_move",
                "city_median_pps",
            ]
        )

    ordinal = ["Furnished_Status", "Public_Transport_Accessibility"]
    onehot = ["Property_Type", "Facing", "Owner_Type", "Availability_Status"]
    high_card = ["State", "City", "Locality"]
    return {
        "numeric": numeric,
        "ordinal": ordinal,
        "onehot": onehot,
        "high_card": high_card,
    }


def build_preprocessor(task: str) -> ColumnTransformer:
    cols = _feature_columns_for_task(task)
    ordinal_categories = [
        ["Unfurnished", "Semi-furnished", "Furnished"],
        ["Low", "Medium", "High"],
    ]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cols["numeric"]),
            (
                "ord",
                Pipeline(
                    [
                        (
                            "enc",
                            OrdinalEncoder(
                                categories=ordinal_categories,
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                        ("sc", StandardScaler()),
                    ]
                ),
                cols["ordinal"],
            ),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cols["onehot"],
            ),
            (
                "te",
                Pipeline(
                    [
                        (
                            "enc",
                            TargetEncoder(smooth="auto", cv=5, random_state=42),
                        ),
                        ("sc", StandardScaler()),
                    ]
                ),
                cols["high_card"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model_pipeline(estimator, task: str) -> Pipeline:
    """Exact preprocessing used in training is reused in the Streamlit app."""
    return Pipeline(
        steps=[
            ("fe", HousingFeatureEngineer()),
            ("pre", build_preprocessor(task)),
            ("model", clone(estimator)),
        ]
    )


def raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in RAW_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")
    return df[RAW_INPUT_COLUMNS].copy()


def unused_columns_note(task: str) -> Iterable[str]:
    return CLASSIFIER_EXCLUDE if task == "classifier" else REGRESSOR_EXCLUDE
