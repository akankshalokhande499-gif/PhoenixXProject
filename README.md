# 🌱 WasteWise AI

Agricultural (sugarcane) waste utilization decision-support app with a real
**machine learning** scoring engine, built with Streamlit + scikit-learn.

Given a waste type, quantity and farmer location, the app:
- Predicts a facility **suitability score (0–100)** using a trained
  `RandomForestRegressor`, instead of a fixed hand-tuned formula
- Shows research-backed utilization methods for that waste type (from `utilization_methods.csv`)
- Plots facilities on an interactive map with the recommended route
- Surfaces real **feature importances** from the trained model so recommendations are explainable
- Summarizes economic and environmental impact

## What makes this an ML project

| Component | Detail |
|---|---|
| **Model** | `RandomForestRegressor` (scikit-learn), wrapped in a `Pipeline` with `StandardScaler` |
| **Training data** | `train_model.py` simulates 6,000 synthetic historical farmer→facility transactions with a nonlinear ground-truth "suitability" outcome (diminishing returns on net value, an accelerating distance penalty past 60km, capacity-fit bonus, transport-cost-burden penalty, plus noise) |
| **Features** | `distance_km, price_per_ton, capacity_tons, quantity_tons, capacity_ratio, net_value_per_ton, transport_cost_ratio, trips` |
| **Evaluation** | 80/20 train/test split; R² and MAE reported and shown live in the app sidebar (`model/metrics.json`) |
| **Explainability** | `feature_importances_` from the trained forest are saved to `model/feature_importance.csv` and rendered as a live bar chart in the app ("What the ML model learned matters most") |
| **Inference** | `calculate_recommendations()` in `app.py` builds a feature row per candidate facility and calls `model.predict()` — this is what ranks facilities, replacing the old fixed-weight formula |

This is a legitimate small-scale supervised regression project: synthetic-but-structured
training data → trained model → saved artifact → loaded at inference time → explainable
via feature importance. It is not just "if/else" business rules anymore (though the old
formula is kept as a safety fallback if the model file is missing).

## Project structure

```
wastewise_ai/
├── app.py                       # Streamlit application (loads model, runs inference)
├── train_model.py               # Generates synthetic training data & trains the RandomForest
├── model/
│   ├── suitability_model.joblib # Trained sklearn Pipeline (scaler + RandomForestRegressor)
│   ├── feature_importance.csv   # Feature importances from the trained model
│   └── metrics.json             # Train/test R² and MAE
├── industries.csv               # Sample facility dataset (name, city, type, waste, capacity, price, lat, lon)
├── utilization_methods.csv      # Research dataset of utilization methods (from your uploaded CSV)
├── requirements.txt             # Python dependencies (incl. scikit-learn, joblib)
└── README.md
```

## Setup

1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Model is pre-trained and included in `model/`.) To retrain from scratch:

   ```bash
   python train_model.py
   ```

   This regenerates `model/suitability_model.joblib`, `model/feature_importance.csv`
   and `model/metrics.json`.

4. Run the app:

   ```bash
   streamlit run app.py
   ```

5. Open the URL Streamlit prints (usually `http://localhost:8501`).

## Data

- **`industries.csv`** — sample facility data across Pune, Satara, Kolhapur and Sangli, covering
  Bagasse, Press Mud and Sugarcane Trash. Columns: `name, city, type, waste, capacity, price, lat, lon`.
  Replace with verified real facility data before real-world deployment.
- **`utilization_methods.csv`** — your uploaded research dataset (24 rows) describing utilization
  methods per waste type (cogeneration, pulp/paper, particleboard, 2G ethanol, composting, biochar,
  bio-fertilizer, biogas, mulching, etc.), including descriptions, processing complexity, cost/revenue
  notes, environmental benefit and source confidence. The app reads this file directly to populate the
  "Possible Utilization Methods" cards — no hardcoded text.

## Notes / next steps for production

- **Replace synthetic training data with real historical transactions** (actual farmer
  choices, deal outcomes, or satisfaction ratings) as soon as you have them — this is the
  single highest-value upgrade. Retrain via `train_model.py` with a real dataset in place
  of `simulate_transactions()`.
- Swap sample facility prices/capacities for verified, up-to-date data.
- Replace the fixed `LOCATIONS` dict with a geocoding lookup or user-entered coordinates.
- Consider persisting analyses (e.g. to a database) if you need history/audit trails, which
  also gives you real labels to retrain the model on over time.
- Try alternative models (Gradient Boosting, XGBoost) or hyperparameter tuning (`GridSearchCV`)
  once real data is available — the current settings are tuned for the synthetic data shape.
- The old fixed-weight formula (`0.35*economic + 0.30*distance + 0.20*capacity + 0.15*price`)
  is retained only as a fallback in `calculate_recommendations()` if `model/suitability_model.joblib`
  is missing.
