"""
train_model.py
================
Generates a synthetic "historical transactions" dataset that mimics past
farmer -> facility waste-sale outcomes, then trains a RandomForestRegressor
to predict a 0-100 facility suitability score from transaction features.

This replaces the old hand-tuned linear formula
(0.35*economic + 0.30*distance + 0.20*capacity + 0.15*price) with a model
that LEARNS the relationship between transaction features and outcome
quality (a proxy for "would the farmer be satisfied / would this deal make
sense") from data, including non-linear effects and feature interactions
a fixed linear formula can't capture.

Run once to (re)train:
    python train_model.py

Produces:
    model/suitability_model.joblib   -> trained sklearn Pipeline
    model/feature_importance.csv     -> feature importances for explainability
    model/metrics.json               -> train/test evaluation metrics
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

RNG = np.random.default_rng(42)
N_SAMPLES = 6000

FEATURES = [
    "distance_km",
    "price_per_ton",
    "capacity_tons",
    "quantity_tons",
    "capacity_ratio",       # accepted_quantity / requested quantity
    "net_value_per_ton",    # (revenue - transport_cost) / quantity
    "transport_cost_ratio", # transport_cost / revenue (cost burden)
    "trips",
]

TARGET = "suitability_score"


def simulate_transactions(n=N_SAMPLES):
    """Simulate realistic past farmer-facility transactions."""
    distance_km = RNG.uniform(1, 180, n)
    price_per_ton = RNG.uniform(800, 4500, n)
    capacity_tons = RNG.uniform(10, 160, n)
    quantity_tons = RNG.uniform(0.5, 200, n)

    accepted_quantity = np.minimum(quantity_tons, capacity_tons)
    capacity_ratio = accepted_quantity / np.maximum(quantity_tons, 1e-6)

    truck_capacity = 5
    trips = np.maximum(1, np.ceil(accepted_quantity / truck_capacity))
    transport_rate = 50
    transport_cost = distance_km * transport_rate * trips
    revenue = accepted_quantity * price_per_ton
    net_value = revenue - transport_cost
    net_value_per_ton = net_value / np.maximum(quantity_tons, 1e-6)
    transport_cost_ratio = transport_cost / np.maximum(revenue, 1e-6)

    # ---- Ground-truth outcome (what we're trying to learn) ----
    # A nonlinear "historical satisfaction / deal quality" score that a
    # simple linear formula would only approximate:
    #  - diminishing returns on net value per ton (sqrt)
    #  - a distance penalty that accelerates after ~60km (nonlinear)
    #  - a bonus if capacity comfortably covers the full quantity
    #  - a penalty when transport cost eats up a large share of revenue
    #  - random noise to mimic real-world unpredictability
    distance_penalty = np.where(
        distance_km <= 60,
        distance_km * 0.35,
        60 * 0.35 + (distance_km - 60) * 0.9
    )
    value_term = 18 * np.sign(net_value_per_ton) * np.sqrt(np.abs(net_value_per_ton) + 1)
    capacity_bonus = np.where(capacity_ratio >= 0.999, 8, capacity_ratio * 5)
    cost_burden_penalty = transport_cost_ratio * 25
    price_term = (price_per_ton / 4500) * 12

    noise = RNG.normal(0, 6, n)

    raw_score = (
        40
        + value_term
        - distance_penalty
        + capacity_bonus
        - cost_burden_penalty
        + price_term
        + noise
    )
    suitability_score = np.clip(raw_score, 0, 100)

    data = pd.DataFrame({
        "distance_km": distance_km,
        "price_per_ton": price_per_ton,
        "capacity_tons": capacity_tons,
        "quantity_tons": quantity_tons,
        "capacity_ratio": capacity_ratio,
        "net_value_per_ton": net_value_per_ton,
        "transport_cost_ratio": transport_cost_ratio,
        "trips": trips,
        TARGET: suitability_score,
    })
    return data


def train():
    df = simulate_transactions()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    metrics = {
        "train_r2": round(r2_score(y_train, train_pred), 4),
        "test_r2": round(r2_score(y_test, test_pred), 4),
        "train_mae": round(mean_absolute_error(y_train, train_pred), 3),
        "test_mae": round(mean_absolute_error(y_test, test_pred), 3),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    importances = pipeline.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": importances
    }).sort_values("importance", ascending=False)

    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, "model/suitability_model.joblib")
    importance_df.to_csv("model/feature_importance.csv", index=False)
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print("\nFeature importances:")
    print(importance_df.to_string(index=False))


if __name__ == "__main__":
    train()
