"""Answer 20 EDA questions with charts and short written insights."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import apply_plot_style, data_processed_path, figures_dir, reports_dir

COLOR = "#2c5282"
COLOR2 = "#718096"


def load_cleaned() -> pd.DataFrame:
    path = data_processed_path()
    if not path.exists():
        raise FileNotFoundError("Run `python -m src.preprocessing` first.")
    return pd.read_csv(path)


def _save(fig: plt.Figure, name: str) -> Path:
    path = figures_dir() / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def q01_price_distribution(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["Price_in_Lakhs"], bins=40, color=COLOR, edgecolor="white")
    ax.set_title("Distribution of property price (₹ lakhs)")
    ax.set_xlabel("Price (lakhs)")
    ax.set_ylabel("Count")
    _save(fig, "q01_price_distribution.png")
    return (
        "List prices span ~10–500 lakhs and are close to uniform, which is typical of "
        "synthetic data rather than a real market with a right-skewed luxury tail."
    )


def q02_size_distribution(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["Size_in_SqFt"], bins=40, color=COLOR, edgecolor="white")
    ax.set_title("Distribution of property size (sq ft)")
    ax.set_xlabel("Size (sq ft)")
    ax.set_ylabel("Count")
    _save(fig, "q02_size_distribution.png")
    return (
        "Sizes fill 500–5000 sq ft almost evenly. There is no clustering around "
        "common builder-floor plates, so size is not a strong city-specific signal here."
    )


def q03_pps_by_property_type(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt_INR", color=COLOR, ax=ax, fliersize=1)
    ax.set_title("Price per sqft (₹) by property type")
    ax.set_ylabel("₹ / sq ft")
    _save(fig, "q03_pps_by_property_type.png")
    med = df.groupby("Property_Type")["Price_per_SqFt_INR"].median().round(0).to_dict()
    return (
        f"Median ₹/sqft is similar across types ({med}). Type does not separate unit value "
        "in this file the way it would in a real listing market."
    )


def q04_size_vs_price(df: pd.DataFrame) -> str:
    sample = df.sample(n=min(8000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sample["Size_in_SqFt"], sample["Price_in_Lakhs"], s=8, alpha=0.25, c=COLOR, linewidths=0)
    ax.set_title("Size vs price (8,000-row sample)")
    ax.set_xlabel("Size (sq ft)")
    ax.set_ylabel("Price (lakhs)")
    _save(fig, "q04_size_vs_price.png")
    corr = float(df["Size_in_SqFt"].corr(df["Price_in_Lakhs"]))
    return (
        f"Pearson correlation between size and list price is {corr:.3f}. Larger homes are "
        "not systematically more expensive, so ₹/sqft varies widely."
    )


def q05_outliers_pps_size(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].boxplot(df["Price_per_SqFt_INR"], flierprops={"marker": ".", "alpha": 0.3})
    axes[0].set_title("Outliers: ₹ / sq ft")
    axes[0].set_ylabel("₹ / sq ft")
    axes[1].boxplot(df["Size_in_SqFt"], flierprops={"marker": ".", "alpha": 0.3})
    axes[1].set_title("Outliers: size (sq ft)")
    _save(fig, "q05_outliers_pps_size.png")
    n_pps = int(df["outlier_pps"].sum())
    n_size = int(df["outlier_size"].sum())
    return (
        f"IQR flags {n_pps:,} ₹/sqft outliers ({df['outlier_pps'].mean()*100:.1f}%) and "
        f"{n_size:,} size outliers. Size stays inside 500–5000; high ₹/sqft is small area + high price. None were dropped."
    )


def q06_avg_pps_by_state(df: pd.DataFrame) -> str:
    stats = df.groupby("State")["Price_per_SqFt_INR"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(stats.index, stats.values, color=COLOR)
    ax.set_title("Average ₹ / sq ft by state")
    ax.set_xlabel("₹ / sq ft")
    _save(fig, "q06_avg_pps_by_state.png")
    return (
        f"State averages sit in a narrow band ({stats.min():.0f}–{stats.max():.0f} ₹/sqft). "
        "Do not treat this as a real interstate price ranking; the source looks synthetic."
    )


def q07_avg_price_by_city(df: pd.DataFrame) -> str:
    stats = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(stats.index, stats.values, color=COLOR)
    ax.set_title("Average list price (lakhs) by city")
    ax.set_xlabel("Lakhs")
    _save(fig, "q07_avg_price_by_city.png")
    return (
        f"City mean prices only range {stats.min():.1f}–{stats.max():.1f} lakhs. "
        "Location barely moves list price in this dataset."
    )


def q08_median_age_top_localities(df: pd.DataFrame) -> str:
    top = df["Locality"].value_counts().head(15).index
    sub = df[df["Locality"].isin(top)]
    stats = sub.groupby("Locality")["Age_recomputed"].median().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(stats.index.astype(str), stats.values, color=COLOR)
    ax.set_title("Median property age — 15 most common localities")
    ax.set_xlabel("Years")
    _save(fig, "q08_median_age_by_locality.png")
    return (
        f"Among the 15 largest localities, median age runs {stats.min():.0f}–{stats.max():.0f} years. "
        "Stock is mixed-age rather than a single construction boom."
    )


def q09_bhk_across_cities(df: pd.DataFrame) -> str:
    top_cities = df["City"].value_counts().head(8).index
    sub = df[df["City"].isin(top_cities)]
    ct = pd.crosstab(sub["City"], sub["BHK"], normalize="index")
    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="Greys")
    ax.set_title("BHK mix in the 8 largest cities")
    ax.set_ylabel("Share")
    ax.legend(title="BHK", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, "q09_bhk_across_cities.png")
    return (
        "BHK 1–5 are spread almost evenly in every large city. Bedroom mix is not a "
        "city differentiator here."
    )


def q10_price_trend_expensive_localities(df: pd.DataFrame) -> str:
    loc_med = df.groupby("Locality")["Price_in_Lakhs"].median().sort_values(ascending=False)
    top5 = loc_med.head(5).index
    sub = df[df["Locality"].isin(top5)]
    trend = sub.groupby(["Year_Built", "Locality"])["Price_in_Lakhs"].median().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    for loc, g in trend.groupby("Locality"):
        ax.plot(g["Year_Built"], g["Price_in_Lakhs"], label=str(loc), linewidth=1.5)
    ax.set_title("Median price vs Year_Built — 5 highest-median localities")
    ax.set_xlabel("Year built (used as the only time axis)")
    ax.set_ylabel("Median price (lakhs)")
    ax.legend(fontsize=8)
    _save(fig, "q10_price_trend_top_localities.png")
    return (
        "There is no listing date in the file, so Year_Built is the time axis. "
        "The five priciest localities do not show a clean vintage-price trend."
    )


def q11_correlation_heatmap(df: pd.DataFrame) -> str:
    cols = [
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Price_per_SqFt_INR",
        "Age_recomputed",
        "Nearby_Schools",
        "Nearby_Hospitals",
        "infrastructure_score",
        "amenity_count",
        "floor_ratio",
        "bhk_per_1000sqft",
        "Good_Investment",
        "Future_Price_5Y_Lakhs",
    ]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, cmap="Greys", center=0, annot=False, ax=ax, square=True)
    ax.set_title("Numeric correlation heatmap")
    _save(fig, "q11_correlation_heatmap.png")
    pps_price = corr.loc["Price_per_SqFt_INR", "Price_in_Lakhs"]
    return (
        f"The strongest structural link is list price vs ₹/sqft (r={pps_price:.2f}) because "
        "₹/sqft is computed from price. Infrastructure and amenities barely correlate with price."
    )


def q12_schools_vs_pps(df: pd.DataFrame) -> str:
    stats = df.groupby("Nearby_Schools")["Price_per_SqFt_INR"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(stats.index, stats.values, marker="o", color=COLOR)
    ax.set_title("Average ₹ / sq ft vs nearby schools")
    ax.set_xlabel("Nearby schools")
    ax.set_ylabel("₹ / sq ft")
    _save(fig, "q12_schools_vs_pps.png")
    return (
        "Average ₹/sqft is essentially flat across school counts. Nearby_Schools is not "
        "priced in this synthetic table."
    )


def q13_hospitals_vs_pps(df: pd.DataFrame) -> str:
    stats = df.groupby("Nearby_Hospitals")["Price_per_SqFt_INR"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(stats.index, stats.values, marker="o", color=COLOR)
    ax.set_title("Average ₹ / sq ft vs nearby hospitals")
    ax.set_xlabel("Nearby hospitals")
    ax.set_ylabel("₹ / sq ft")
    _save(fig, "q13_hospitals_vs_pps.png")
    return (
        "Hospital count also shows no material ₹/sqft gradient. Do not infer a healthcare-premium from this file."
    )


def q14_price_by_furnished(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = ["Unfurnished", "Semi-furnished", "Furnished"]
    sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs", order=order, color=COLOR, ax=ax, fliersize=1)
    ax.set_title("Price (lakhs) by furnished status")
    _save(fig, "q14_price_by_furnished.png")
    med = df.groupby("Furnished_Status")["Price_in_Lakhs"].median().round(1).to_dict()
    return (
        f"Median prices by furnishing: {med}. The three boxes overlap heavily; furnishing is not a price driver here."
    )


def q15_pps_by_facing(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=df, x="Facing", y="Price_per_SqFt_INR", color=COLOR, ax=ax, fliersize=1)
    ax.set_title("₹ / sq ft by facing")
    ax.set_ylabel("₹ / sq ft")
    _save(fig, "q15_pps_by_facing.png")
    return (
        "North/South/East/West have indistinguishable ₹/sqft distributions. Facing has no pricing role in this data."
    )


def q16_count_by_owner(df: pd.DataFrame) -> str:
    counts = df["Owner_Type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(counts.index.astype(str), counts.values, color=COLOR)
    ax.set_title("Listings by owner type")
    ax.set_ylabel("Count")
    _save(fig, "q16_count_by_owner.png")
    return (
        f"Owner/Builder/Broker counts are nearly equal ({counts.to_dict()}). Supply is balanced by construction, not organic."
    )


def q17_count_by_availability(df: pd.DataFrame) -> str:
    counts = df["Availability_Status"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(counts.index.astype(str), counts.values, color=COLOR)
    ax.set_title("Listings by availability")
    ax.set_ylabel("Count")
    _save(fig, "q17_count_by_availability.png")
    return (
        f"Ready_to_Move vs Under_Construction is a near 50/50 split ({counts.to_dict()})."
    )


def q18_parking_vs_price(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.boxplot(data=df, x="Parking_Space", y="Price_in_Lakhs", color=COLOR, ax=ax, fliersize=1)
    ax.set_title("Price (lakhs) by parking")
    _save(fig, "q18_parking_vs_price.png")
    med = df.groupby("Parking_Space")["Price_in_Lakhs"].median().round(1).to_dict()
    return (
        f"Median price with vs without parking: {med}. Parking does not lift list price in this table."
    )


def q19_amenities_vs_pps(df: pd.DataFrame) -> str:
    stats = df.groupby("amenity_count")["Price_per_SqFt_INR"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(stats.index.astype(int), stats.values, color=COLOR)
    ax.set_title("Average ₹ / sq ft by amenity count")
    ax.set_xlabel("Amenity count")
    ax.set_ylabel("₹ / sq ft")
    _save(fig, "q19_amenities_vs_pps.png")
    return (
        "Mean ₹/sqft is flat across amenity counts. Amenities were generated independently of price."
    )


def q20_transport_pps_and_good_inv(df: pd.DataFrame) -> str:
    order = ["Low", "Medium", "High"]
    pps = df.groupby("Public_Transport_Accessibility")["Price_per_SqFt_INR"].mean().reindex(order)
    gi = df.groupby("Public_Transport_Accessibility")["Good_Investment"].mean().reindex(order)
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(order))
    ax1.bar(x - 0.18, pps.values, width=0.36, color=COLOR, label="₹ / sq ft")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, gi.values, width=0.36, color=COLOR2, label="Good-investment rate")
    ax1.set_xticks(x, order)
    ax1.set_ylabel("₹ / sq ft")
    ax2.set_ylabel("Good-investment rate")
    ax1.set_title("Public transport vs ₹/sqft and good-investment rate")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    _save(fig, "q20_transport_pps_good_investment.png")
    return (
        f"₹/sqft is similar across transport levels, but the good-investment rate rises with High transport "
        f"({gi.round(3).to_dict()}) because High transport is one of the six labelling rules."
    )


QUESTIONS = [
    ("1. Price distribution", q01_price_distribution),
    ("2. Size distribution", q02_size_distribution),
    ("3. Price per sqft by property type", q03_pps_by_property_type),
    ("4. Size vs price", q04_size_vs_price),
    ("5. Outliers in INR/sqft and size", q05_outliers_pps_size),
    ("6. Average INR/sqft by state", q06_avg_pps_by_state),
    ("7. Average price by city", q07_avg_price_by_city),
    ("8. Median age by locality (top 15)", q08_median_age_top_localities),
    ("9. BHK mix across cities", q09_bhk_across_cities),
    ("10. Price trend for top 5 expensive localities (Year_Built)", q10_price_trend_expensive_localities),
    ("11. Numeric correlation heatmap", q11_correlation_heatmap),
    ("12. Schools vs INR/sqft", q12_schools_vs_pps),
    ("13. Hospitals vs INR/sqft", q13_hospitals_vs_pps),
    ("14. Price by furnished status", q14_price_by_furnished),
    ("15. INR/sqft by facing", q15_pps_by_facing),
    ("16. Count by owner type", q16_count_by_owner),
    ("17. Count by availability", q17_count_by_availability),
    ("18. Parking vs price", q18_parking_vs_price),
    ("19. Amenities vs INR/sqft", q19_amenities_vs_pps),
    ("20. Transport vs INR/sqft and good-investment rate", q20_transport_pps_and_good_inv),
]


def run() -> Path:
    apply_plot_style()
    sns.set_theme(style="white", rc={"axes.grid": False})
    df = load_cleaned()
    lines = [
        "# EDA insights (20 questions)",
        "",
        "Charts are saved under `reports/figures/`. Prices in this file are synthetic and similar across cities; insights describe the table, not the Indian housing market.",
        "",
    ]
    for title, fn in QUESTIONS:
        print(f"plotting {title}")
        insight = fn(df)
        lines.extend([f"## {title}", insight, ""])
    out = reports_dir() / "eda_insights.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    run()
