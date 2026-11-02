"""Streamlit UI: predict, explore, and review EDA insights."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.evaluate import extract_feature_importance
from src.features import engineer_frame
from src.utils import (
    AMENITY_VOCAB,
    APPRECIATION_CONFIG,
    GOOD_INVESTMENT_THRESHOLD,
    apply_plot_style,
    city_tier,
    data_processed_path,
    figures_dir,
    models_dir,
)

st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")
apply_plot_style()


@st.cache_data(show_spinner=False)
def load_cleaned() -> pd.DataFrame:
    path = data_processed_path()
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_models(model_signature: tuple[int | None, int | None]):
    clf_path = models_dir() / "good_investment_pipeline.joblib"
    reg_path = models_dir() / "future_price_pipeline.joblib"
    clf = joblib.load(clf_path) if clf_path.exists() else None
    reg = joblib.load(reg_path) if reg_path.exists() else None
    return clf, reg


def form_to_frame(values: dict) -> pd.DataFrame:
    amenities = ", ".join(values["amenities"])
    return pd.DataFrame(
        [
            {
                "State": values["state"],
                "City": values["city"],
                "Locality": values["locality"],
                "Property_Type": values["property_type"],
                "BHK": values["bhk"],
                "Size_in_SqFt": values["size"],
                "Price_in_Lakhs": values["price"],
                "Year_Built": values["year_built"],
                "Furnished_Status": values["furnished"],
                "Floor_No": values["floor_no"],
                "Total_Floors": values["total_floors"],
                "Nearby_Schools": values["schools"],
                "Nearby_Hospitals": values["hospitals"],
                "Public_Transport_Accessibility": values["transport"],
                "Parking_Space": values["parking"],
                "Security": values["security"],
                "Amenities": amenities,
                "Facing": values["facing"],
                "Owner_Type": values["owner"],
                "Availability_Status": values["availability"],
            }
        ]
    )


def rule_based_prediction(row: pd.DataFrame, df: pd.DataFrame) -> tuple[int, float, int]:
    city_stats = df.groupby("City").agg(
        city_median_price=("Price_in_Lakhs", "median"),
        city_median_pps=("Price_per_SqFt_INR", "median"),
    )
    features, _ = engineer_frame(row, city_stats=city_stats)
    property_row = features.iloc[0]
    score = sum(
        [
            int(property_row["Price_per_SqFt_INR"] <= property_row["city_median_pps"]),
            int(property_row["BHK"] >= 3),
            int(property_row["is_ready_to_move"]),
            int(property_row["infrastructure_score"] > df["infrastructure_score"].median()),
            int(property_row["amenity_count"] > df["amenity_count"].median()),
            int(property_row["Public_Transport_Accessibility"] == "High"),
        ]
    )

    config = APPRECIATION_CONFIG
    rate = (
        config["tier_base"][city_tier(str(property_row["City"]))]
        + config["property_type_adj"].get(str(property_row["Property_Type"]), 0.0)
        + property_row["infrastructure_score"] * config["infra_adj_max"]
    )
    rate = float(np.clip(rate, config["min_rate"], config["max_rate"]))
    future_price = float(property_row["Price_in_Lakhs"] * (1 + rate) ** config["horizon_years"])
    return int(score >= GOOD_INVESTMENT_THRESHOLD), future_price, score


def page_predict(df: pd.DataFrame) -> None:
    st.header("Predict")
    st.caption("Good-investment label and estimated price after 5 years, using saved models when available and documented assumptions otherwise.")
    classifier_path = models_dir() / "good_investment_pipeline.joblib"
    regressor_path = models_dir() / "future_price_pipeline.joblib"
    model_signature = tuple(
        path.stat().st_mtime_ns if path.exists() else None
        for path in (classifier_path, regressor_path)
    )
    clf, reg = load_models(model_signature)
    if clf is None or reg is None:
        st.info("Saved model pipelines are not available yet. Missing predictions use the project's documented rule-based assumptions.")

    states = sorted(df["State"].unique()) if len(df) else []
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        state = st.selectbox("State", states)
    cities = sorted(df.loc[df["State"] == state, "City"].unique()) if len(df) else []
    with col_b:
        city = st.selectbox("City", cities)
    localities = sorted(df.loc[df["City"] == city, "Locality"].unique()) if len(df) else []
    with col_c:
        locality = st.selectbox("Locality", localities)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        property_type = st.selectbox("Property Type", ["Apartment", "Independent House", "Villa"])
        bhk = st.slider("BHK", 1, 5, 2)
        size = st.number_input("Size (sq ft)", min_value=200, max_value=8000, value=1200, step=50)
    with c2:
        price = st.number_input("Price (lakhs)", min_value=5.0, max_value=800.0, value=85.0, step=1.0)
        year_built = st.slider("Year Built", 1990, 2025, 2015)
        furnished = st.selectbox("Furnished", ["Unfurnished", "Semi-furnished", "Furnished"])
    with c3:
        floor_no = st.number_input("Floor No", min_value=0, max_value=50, value=3)
        total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=10)
        schools = st.slider("Nearby Schools", 1, 10, 5)
        hospitals = st.slider("Nearby Hospitals", 1, 10, 4)
    with c4:
        transport = st.selectbox("Public Transport", ["Low", "Medium", "High"])
        parking = st.selectbox("Parking", ["Yes", "No"])
        security = st.selectbox("Security", ["Yes", "No"])
        facing = st.selectbox("Facing", ["North", "South", "East", "West"])
        owner = st.selectbox("Owner Type", ["Owner", "Builder", "Broker"])
        availability = st.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])

    amenities = st.multiselect("Amenities", AMENITY_VOCAB, default=["Gym"])

    if st.button("Run prediction", type="primary"):
        row = form_to_frame(
            {
                "state": state,
                "city": city,
                "locality": locality,
                "property_type": property_type,
                "bhk": int(bhk),
                "size": int(size),
                "price": float(price),
                "year_built": int(year_built),
                "furnished": furnished,
                "floor_no": int(floor_no),
                "total_floors": int(total_floors),
                "schools": int(schools),
                "hospitals": int(hospitals),
                "transport": transport,
                "parking": parking,
                "security": security,
                "amenities": amenities,
                "facing": facing,
                "owner": owner,
                "availability": availability,
            }
        )
        fallback_pred, fallback_future, fallback_score = rule_based_prediction(row, df)
        pred = int(clf.predict(row)[0]) if clf is not None else fallback_pred
        proba = float(clf.predict_proba(row)[0, 1]) if clf is not None else None
        future = float(reg.predict(row)[0]) if reg is not None else fallback_future
        appreciation_pct = (future / float(price) - 1.0) * 100 if price else 0.0

        m1, m2, m3 = st.columns(3)
        investment_delta = f"confidence {proba:.1%}" if proba is not None else f"rule score {fallback_score}/6"
        m1.metric("Good Investment", "Yes" if pred == 1 else "No", investment_delta)
        m2.metric("Estimated price after 5 years", f"{future:.2f} lakhs")
        m3.metric("Expected appreciation", f"{appreciation_pct:.1f}%")

        names, values = extract_feature_importance(clf) if clf is not None else (None, None)
        if names is not None:
            order = np.argsort(np.abs(values))[-15:]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(np.array(names)[order], np.array(values)[order], color="#2c5282")
            ax.set_title("Classifier feature importance (model-level)")
            st.pyplot(fig)
            plt.close(fig)


def page_explore(df: pd.DataFrame) -> None:
    st.header("Explore")
    if df.empty:
        st.warning("Processed data not found. Run `python -m src.preprocessing`.")
        return

    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        state = st.multiselect("State", sorted(df["State"].unique()))
    with f2:
        city_opts = sorted(df.loc[df["State"].isin(state), "City"].unique()) if state else sorted(df["City"].unique())
        city = st.multiselect("City", city_opts)
    with f3:
        bhk = st.multiselect("BHK", sorted(df["BHK"].unique()))
    with f4:
        ptype = st.multiselect("Property Type", sorted(df["Property_Type"].unique()))
    with f5:
        pmin, pmax = float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max())
        price_range = st.slider("Price (lakhs)", min_value=pmin, max_value=pmax, value=(pmin, pmax))

    view = df.copy()
    if state:
        view = view[view["State"].isin(state)]
    if city:
        view = view[view["City"].isin(city)]
    if bhk:
        view = view[view["BHK"].isin(bhk)]
    if ptype:
        view = view[view["Property_Type"].isin(ptype)]
    view = view[(view["Price_in_Lakhs"] >= price_range[0]) & (view["Price_in_Lakhs"] <= price_range[1])]

    st.caption(f"{len(view):,} listings after filters (of {len(df):,}).")
    show_cols = [
        "State",
        "City",
        "Locality",
        "Property_Type",
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Price_per_SqFt_INR",
        "Year_Built",
        "Good_Investment",
        "Future_Price_5Y_Lakhs",
    ]
    st.dataframe(view[show_cols].head(500), use_container_width=True)

    sample = view.sample(n=min(6000, len(view)), random_state=42) if len(view) else view
    c1, c2 = st.columns(2)
    with c1:
        city_price = view.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.barh(city_price.index.astype(str), city_price.values, color="#2c5282")
        ax.set_title("Average price by city (filtered)")
        ax.set_xlabel("Lakhs")
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(sample["Size_in_SqFt"], sample["Price_in_Lakhs"], s=8, alpha=0.25, c="#2c5282", linewidths=0)
        ax.set_title("Price vs size")
        ax.set_xlabel("Sq ft")
        ax.set_ylabel("Lakhs")
        st.pyplot(fig)
        plt.close(fig)

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        view.boxplot(column="Price_per_SqFt_INR", by="Property_Type", ax=ax, grid=False)
        ax.set_title("₹ / sq ft by property type")
        ax.set_xlabel("")
        plt.suptitle("")
        st.pyplot(fig)
        plt.close(fig)
    with c4:
        trend = view.groupby("Year_Built")["Price_in_Lakhs"].median()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(trend.index, trend.values, color="#2c5282")
        ax.set_title("Median price by Year_Built")
        ax.set_xlabel("Year built")
        ax.set_ylabel("Lakhs")
        st.pyplot(fig)
        plt.close(fig)


def page_insights(df: pd.DataFrame) -> None:
    st.header("Insights")
    st.caption("Prices in this file are synthetic and similar across cities. Charts describe the table, not the real Indian market.")
    if df.empty:
        st.warning("Processed data not found.")
        return

    heat = df.pivot_table(
        index="State",
        columns="Property_Type",
        values="Price_per_SqFt_INR",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(heat.values, aspect="auto", cmap="Greys")
    ax.set_xticks(range(len(heat.columns)), list(heat.columns), rotation=30, ha="right")
    ax.set_yticks(range(len(heat.index)), list(heat.index))
    ax.set_title("Average ₹ / sq ft by state and property type")
    fig.colorbar(im, ax=ax, fraction=0.03)
    st.pyplot(fig)
    plt.close(fig)

    gi = df.groupby("Public_Transport_Accessibility")["Good_Investment"].mean().reindex(["Low", "Medium", "High"])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(gi.index.astype(str), gi.values, color="#2c5282")
    ax.set_ylim(0, 1)
    ax.set_title("Good-investment rate by public transport")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Saved EDA charts")
    picks = [
        "q11_correlation_heatmap.png",
        "q06_avg_pps_by_state.png",
        "q04_size_vs_price.png",
        "q20_transport_pps_good_investment.png",
        "q10_price_trend_top_localities.png",
        "q05_outliers_pps_size.png",
    ]
    cols = st.columns(2)
    for i, name in enumerate(picks):
        path = figures_dir() / name
        if path.exists():
            cols[i % 2].image(str(path), caption=name)

    insight_path = ROOT / "reports" / "eda_insights.md"
    if insight_path.exists():
        st.markdown(insight_path.read_text(encoding="utf-8"))


def main() -> None:
    st.sidebar.title("Real Estate Investment Advisor")
    page = st.sidebar.radio("Page", ["Predict", "Explore", "Insights"])
    df = load_cleaned()
    if page == "Predict":
        page_predict(df)
    elif page == "Explore":
        page_explore(df)
    else:
        page_insights(df)


if __name__ == "__main__":
    main()
