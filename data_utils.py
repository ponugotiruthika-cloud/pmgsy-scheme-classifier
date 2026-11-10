"""
PMGSY Dataset Loader and Preprocessor
Handles loading from local CSV or generating realistic synthetic data
matching the AI Kosh PMGSY dataset structure.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

# ─────────────────────────────────────────────
# PMGSY scheme labels (target variable)
# ─────────────────────────────────────────────
SCHEME_LABELS = [
    "PMGSY-I",
    "PMGSY-II",
    "RCPLWEA",
    "PMGSY-III",
    "MNREGA",
]

# ─────────────────────────────────────────────
# Feature columns used for training
# ─────────────────────────────────────────────
NUMERIC_FEATURES = [
    "ROAD_LENGTH_KM",
    "BRIDGE_LENGTH_MTR",
    "DPR_COST_LAKH",
    "REVISED_COST_LAKH",
    "EXPENDITURE_LAKH",
    "HABITATION_POPULATION",
    "CONNECTED_HABITATION",
    "NUM_PACKAGES",
]

CATEGORICAL_FEATURES = [
    "STATE_NAME",
    "PROJECT_TYPE",     # Road / Bridge
    "SURFACE_TYPE",     # BT / CC / WBM / Gravel
    "TERRAIN_TYPE",     # Plain / Rolling / Hilly / Steep
    "SANCTION_YEAR",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "PMGSY_SCHEME"


# ─────────────────────────────────────────────
# Synthetic data generator
# ─────────────────────────────────────────────

STATE_NAMES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chhattisgarh", "Gujarat", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal",
]

PROJECT_TYPES = ["Road", "Bridge"]
SURFACE_TYPES = ["BT", "CC", "WBM", "Gravel"]
TERRAIN_TYPES = ["Plain", "Rolling", "Hilly", "Steep"]

# Scheme-specific parameter distributions (realistic per-scheme profiles)
SCHEME_PROFILES = {
    "PMGSY-I": {
        "sanction_years": range(2000, 2013),
        "road_len_range": (0.5, 25.0),
        "bridge_len_range": (0, 150),
        "cost_per_km": (10, 50),          # lakh per km
        "pop_range": (100, 500),
        "project_type_weights": [0.88, 0.12],
        "surface_weights": [0.55, 0.10, 0.25, 0.10],
        "terrain_weights": [0.50, 0.25, 0.20, 0.05],
    },
    "PMGSY-II": {
        "sanction_years": range(2013, 2020),
        "road_len_range": (1.0, 30.0),
        "bridge_len_range": (0, 200),
        "cost_per_km": (25, 80),
        "pop_range": (200, 1000),
        "project_type_weights": [0.82, 0.18],
        "surface_weights": [0.65, 0.15, 0.15, 0.05],
        "terrain_weights": [0.45, 0.30, 0.20, 0.05],
    },
    "RCPLWEA": {
        "sanction_years": range(2016, 2024),
        "road_len_range": (0.5, 20.0),
        "bridge_len_range": (0, 100),
        "cost_per_km": (15, 60),
        "pop_range": (50, 300),
        "project_type_weights": [0.80, 0.20],
        "surface_weights": [0.40, 0.05, 0.35, 0.20],
        "terrain_weights": [0.20, 0.25, 0.35, 0.20],   # more hilly
    },
    "PMGSY-III": {
        "sanction_years": range(2019, 2025),
        "road_len_range": (1.0, 35.0),
        "bridge_len_range": (0, 250),
        "cost_per_km": (35, 120),
        "pop_range": (300, 2000),
        "project_type_weights": [0.78, 0.22],
        "surface_weights": [0.70, 0.20, 0.08, 0.02],
        "terrain_weights": [0.40, 0.30, 0.25, 0.05],
    },
    "MNREGA": {
        "sanction_years": range(2006, 2024),
        "road_len_range": (0.1, 8.0),
        "bridge_len_range": (0, 50),
        "cost_per_km": (3, 20),            # lower cost
        "pop_range": (50, 400),
        "project_type_weights": [0.92, 0.08],
        "surface_weights": [0.20, 0.05, 0.45, 0.30],  # more gravel/WBM
        "terrain_weights": [0.55, 0.25, 0.15, 0.05],
    },
}


def _generate_synthetic_data(n_samples: int = 8000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic PMGSY-like records with scheme-specific distributions."""
    rng = np.random.default_rng(random_state)
    records = []

    samples_per_scheme = n_samples // len(SCHEME_LABELS)
    remainder = n_samples % len(SCHEME_LABELS)

    for idx, scheme in enumerate(SCHEME_LABELS):
        p = SCHEME_PROFILES[scheme]
        n = samples_per_scheme + (1 if idx < remainder else 0)
        yrs = list(p["sanction_years"])

        road_len = rng.uniform(*p["road_len_range"], size=n).round(2)
        bridge_len = rng.choice(
            [0, 0, 0] + list(range(10, int(p["bridge_len_range"][1]) + 10, 10)),
            size=n
        ).astype(float)

        cost_per_km = rng.uniform(*p["cost_per_km"], size=n)
        dpr_cost = (road_len * cost_per_km).round(2)
        revised_cost = (dpr_cost * rng.uniform(0.95, 1.30, size=n)).round(2)
        expenditure = (revised_cost * rng.uniform(0.40, 1.05, size=n)).clip(0, revised_cost * 1.1).round(2)

        pop = rng.integers(*p["pop_range"], size=n)
        connected = rng.integers(1, 5, size=n)
        packages = rng.integers(1, 4, size=n)

        states = rng.choice(STATE_NAMES, size=n)
        proj_types = rng.choice(PROJECT_TYPES, size=n, p=p["project_type_weights"])
        surfaces = rng.choice(SURFACE_TYPES, size=n, p=p["surface_weights"])
        terrains = rng.choice(TERRAIN_TYPES, size=n, p=p["terrain_weights"])
        sanction_years = rng.choice(yrs, size=n).astype(str)

        for i in range(n):
            records.append({
                "ROAD_LENGTH_KM": road_len[i],
                "BRIDGE_LENGTH_MTR": bridge_len[i],
                "DPR_COST_LAKH": dpr_cost[i],
                "REVISED_COST_LAKH": revised_cost[i],
                "EXPENDITURE_LAKH": expenditure[i],
                "HABITATION_POPULATION": int(pop[i]),
                "CONNECTED_HABITATION": int(connected[i]),
                "NUM_PACKAGES": int(packages[i]),
                "STATE_NAME": states[i],
                "PROJECT_TYPE": proj_types[i],
                "SURFACE_TYPE": surfaces[i],
                "TERRAIN_TYPE": terrains[i],
                "SANCTION_YEAR": sanction_years[i],
                TARGET: scheme,
            })

    df = pd.DataFrame(records).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def load_data(csv_path: str = None) -> pd.DataFrame:
    """
    Load data from CSV if available, otherwise generate synthetic data.
    Expects the CSV to contain columns matching ALL_FEATURES + TARGET.
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Ensure required columns exist
        missing = [c for c in ALL_FEATURES + [TARGET] if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        return df
    return _generate_synthetic_data()


def preprocess(df: pd.DataFrame):
    """
    Encode categoricals, scale numerics.
    Returns: X_train, X_test, y_train, y_test, label_encoders, scaler, feature_names
    """
    df = df.copy()

    # Encode target
    le_target = LabelEncoder()
    df[TARGET] = le_target.fit_transform(df[TARGET])

    # Encode categoricals
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    X = df[ALL_FEATURES].values
    y = df[TARGET].values

    # Scale numerics (first N columns are numeric)
    scaler = StandardScaler()
    X[:, : len(NUMERIC_FEATURES)] = scaler.fit_transform(X[:, : len(NUMERIC_FEATURES)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, label_encoders, le_target, scaler, ALL_FEATURES
