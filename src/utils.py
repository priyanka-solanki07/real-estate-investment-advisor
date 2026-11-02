"""Paths, constants, and shared helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RANDOM_STATE = 42
TEST_SIZE = 0.20
REFERENCE_YEAR = 2025  # Age_of_Property in the raw file equals 2025 - Year_Built

GOOD_INVESTMENT_THRESHOLD = 3

# Future price: Future = Current * (1 + r) ** 5
# r is NOT a single national rate. It is assembled from city tier + property type +
# infrastructure. These rates are assumptions for a synthetic dataset, not market forecasts.
APPRECIATION_CONFIG = {
    "horizon_years": 5,
    "tier_base": {"1": 0.10, "2": 0.08, "3": 0.06},
    "property_type_adj": {
        "Apartment": 0.000,
        "Independent House": 0.008,
        "Villa": 0.015,
    },
    "infra_adj_max": 0.020,  # added as infrastructure_score * this value (score is 0-1)
    "min_rate": 0.050,
    "max_rate": 0.130,
}

# City tiers are based on conventional Indian metro grouping, NOT on prices in this
# file (prices are nearly identical across cities — see docs/findings.md).
TIER_1_CITIES = {
    "Mumbai",
    "New Delhi",
    "Dwarka",
    "Bangalore",
    "Hyderabad",
    "Chennai",
    "Kolkata",
    "Pune",
    "Ahmedabad",
    "Gurgaon",
    "Noida",
}
TIER_2_CITIES = {
    "Jaipur",
    "Lucknow",
    "Kochi",
    "Surat",
    "Indore",
    "Nagpur",
    "Coimbatore",
    "Patna",
    "Bhopal",
    "Ludhiana",
    "Vadodara",
    "Faridabad",
    "Trivandrum",
    "Visakhapatnam",
    "Vishakhapatnam",
    "Vijayawada",
    "Mysore",
    "Ranchi",
    "Raipur",
    "Guwahati",
    "Dehradun",
    "Jodhpur",
    "Bhubaneswar",
}

TRANSPORT_ORDINAL = {"Low": 1, "Medium": 2, "High": 3}
FURNISHED_ORDINAL = {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2}

AMENITY_VOCAB = ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]

# Leakage control — see docs/methodology.md
# The Good Investment label is a deterministic 6-point rule. We drop the exact
# engineered ingredients of that rule so the classifier is not a lookup of the label.
CLASSIFIER_EXCLUDE = [
    "ID",
    "Amenities",
    "Price_per_SqFt",  # original unreliable lakhs/sqft
    "Good_Investment",
    "investment_score",
    "Future_Price_5Y_Lakhs",
    "appreciation_rate",
    "city_tier",
    "is_ready_to_move",
    "amenity_count",
    "infrastructure_score",
    "city_median_pps",
    "Age_of_Property",
    "Floor_No",
    "Price_per_SqFt_original",
]

# Future price is Current * (1+r)^5 with r from tier / type / infra.
# We still pass current price (needed at inference) but drop the rate itself,
# the other task's target, and city_tier (an exact rate-table key).
REGRESSOR_EXCLUDE = [
    "ID",
    "Amenities",
    "Price_per_SqFt",
    "Good_Investment",
    "investment_score",
    "Future_Price_5Y_Lakhs",
    "appreciation_rate",
    "city_tier",
    "Age_of_Property",
    "Floor_No",
    "Price_per_SqFt_original",
]

RAW_INPUT_COLUMNS = [
    "State",
    "City",
    "Locality",
    "Property_Type",
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Year_Built",
    "Furnished_Status",
    "Floor_No",
    "Total_Floors",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Public_Transport_Accessibility",
    "Parking_Space",
    "Security",
    "Amenities",
    "Facing",
    "Owner_Type",
    "Availability_Status",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_raw_path() -> Path:
    return project_root() / "data" / "raw" / "india_housing_prices.csv"


def data_processed_path() -> Path:
    return project_root() / "data" / "processed" / "cleaned_housing.csv"


def reports_dir() -> Path:
    return project_root() / "reports"


def figures_dir() -> Path:
    path = reports_dir() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = project_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def city_tier(city: str) -> str:
    if city in TIER_1_CITIES:
        return "1"
    if city in TIER_2_CITIES:
        return "2"
    return "3"


def iqr_outlier_mask(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "figure.dpi": 120,
        }
    )


def yes_no_to_int(series: pd.Series) -> pd.Series:
    mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0, True: 1, False: 0, 1: 1, 0: 0}
    return series.map(mapping).fillna(0).astype(int)


def parse_amenities(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]
