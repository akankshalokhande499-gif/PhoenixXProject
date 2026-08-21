import math
import json
import os

import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="WasteWise AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SESSION STATE
# ============================================================
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "analysis_waste_type" not in st.session_state:
    st.session_state.analysis_waste_type = None
if "analysis_quantity" not in st.session_state:
    st.session_state.analysis_quantity = None
if "analysis_city" not in st.session_state:
    st.session_state.analysis_city = None

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
    .stApp { background-color: #f5f8f6; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
    h1, h2, h3 { color: #123d2a; }
    .section-title { font-size: 25px; font-weight: 750; color: #123d2a; margin-top: 30px; margin-bottom: 15px; }
    .section-subtitle { color: #64748b; font-size: 14px; margin-top: -8px; margin-bottom: 20px; }
    .input-card { background: white; padding: 25px; border-radius: 18px; border: 1px solid #e2e8e5;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.04); margin-bottom: 25px; }
    .metric-card { background: white; border-radius: 18px; padding: 22px; border: 1px solid #e2e8e5;
        min-height: 135px; box-shadow: 0 5px 18px rgba(0, 0, 0, 0.04); }
    .metric-icon { font-size: 25px; margin-bottom: 8px; }
    .metric-label { color: #64748b; font-size: 13px; font-weight: 600; }
    .metric-value { color: #123d2a; font-size: 27px; font-weight: 800; margin-top: 5px; }
    .recommendation-card { background: linear-gradient(135deg, #e9f8ef, #ffffff); border: 2px solid #74c69d;
        border-radius: 22px; padding: 28px; margin: 20px 0 25px 0; box-shadow: 0 8px 25px rgba(45, 139, 91, 0.12); }
    .recommendation-label { color: #2d8b5b; font-size: 13px; font-weight: 800; letter-spacing: 1px; }
    .recommendation-title { color: #123d2a; font-size: 30px; font-weight: 800; margin-top: 7px; }
    .recommendation-facility { color: #52665c; font-size: 16px; margin-top: 4px; }
    .score-box { background: #123d2a; color: white; border-radius: 18px; padding: 25px; text-align: center; height: 100%; }
    .score-number { font-size: 42px; font-weight: 800; }
    .score-label { font-size: 13px; opacity: 0.8; }
    .why-card { background: white; border-radius: 18px; padding: 24px; border: 1px solid #e2e8e5;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.04); }
    .reason { padding: 10px 0; border-bottom: 1px solid #edf2ef; color: #334155; font-size: 15px; }
    .reason:last-child { border-bottom: none; }
    .method-card { background: white; border: 1px solid #e2e8e5; border-radius: 18px; padding: 20px;
        min-height: 230px; box-shadow: 0 5px 18px rgba(0, 0, 0, 0.04); }
    .method-name { font-size: 18px; font-weight: 750; color: #123d2a; margin-bottom: 10px; }
    .method-description { font-size: 14px; color: #64748b; line-height: 1.5; min-height: 90px; }
    .potential-high { color: #16803c; font-weight: 800; font-size: 13px; }
    .potential-medium { color: #b7791f; font-weight: 800; font-size: 13px; }
    .potential-low { color: #b91c1c; font-weight: 800; font-size: 13px; }
    .info-card { background: white; border-radius: 18px; border: 1px solid #e2e8e5; padding: 24px;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.04); }
    .environment-card { background: linear-gradient(135deg, #eef9f1, #ffffff); border: 1px solid #b7dfc5;
        border-radius: 18px; padding: 24px; }
    .environment-number { font-size: 30px; font-weight: 800; color: #16803c; }
    .footer { text-align: center; color: #718096; padding: 35px 0 10px 0; font-size: 13px; }
    [data-testid="stSidebar"] { background-color: #f0f6f2; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #123d2a; }
    .stButton > button { border-radius: 12px; min-height: 48px; font-weight: 700; }
    [data-testid="stDataFrame"] { border-radius: 15px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SAMPLE LOCATIONS
# ============================================================
LOCATIONS = {
    "Pune": (18.5204, 73.8567),
    "Satara": (17.6805, 74.0183),
    "Kolhapur": (16.7050, 74.2433),
    "Sangli": (16.8524, 74.5815),
}

# ============================================================
# LOAD FACILITY DATA
# ============================================================
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("industries.csv")
    except FileNotFoundError:
        st.error("industries.csv was not found. Please keep industries.csv in the same folder as app.py.")
        st.stop()
    required_columns = ["name", "city", "type", "waste", "capacity", "price", "lat", "lon"]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        st.error("industries.csv is missing these columns: " + ", ".join(missing))
        st.stop()
    return data


@st.cache_data
def load_methods():
    try:
        data = pd.read_csv("utilization_methods.csv")
    except FileNotFoundError:
        return pd.DataFrame()
    return data


@st.cache_resource
def load_model():
    """Load the trained ML suitability model (RandomForest pipeline)."""
    model_path = "model/suitability_model.joblib"
    importance_path = "model/feature_importance.csv"
    metrics_path = "model/metrics.json"

    if not os.path.exists(model_path):
        return None, None, None

    model = joblib.load(model_path)
    importance_df = pd.read_csv(importance_path) if os.path.exists(importance_path) else None
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return model, importance_df, metrics


df = load_data()
methods_df = load_methods()
ml_model, feature_importance_df, model_metrics = load_model()

ML_FEATURES = [
    "distance_km",
    "price_per_ton",
    "capacity_tons",
    "quantity_tons",
    "capacity_ratio",
    "net_value_per_ton",
    "transport_cost_ratio",
    "trips",
]

# ============================================================
# DISTANCE CALCULATION
# ============================================================
def haversine(lat1, lon1, lat2, lon2):
    radius = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))

# ============================================================
# RECOMMENDATION ENGINE
# ============================================================
def calculate_recommendations(waste_type, quantity, city):
    farmer_lat, farmer_lon = LOCATIONS[city]
    TRUCK_CAPACITY_TONS = 5
    TRANSPORT_RATE_PER_KM = 50

    candidates = df[df["waste"].astype(str).str.lower() == waste_type.lower()].copy()
    if candidates.empty:
        return pd.DataFrame()

    rows = []
    matching_prices = candidates["price"].astype(float)
    max_price = max(float(matching_prices.max()), 1)

    for _, row in candidates.iterrows():
        distance = haversine(farmer_lat, farmer_lon, float(row["lat"]), float(row["lon"]))
        facility_capacity = float(row["capacity"])
        accepted_quantity = min(quantity, facility_capacity)
        price = float(row["price"])
        revenue = accepted_quantity * price
        trips = max(1, math.ceil(accepted_quantity / TRUCK_CAPACITY_TONS))
        transport_cost = distance * TRANSPORT_RATE_PER_KM * trips
        net_value = revenue - transport_cost

        capacity_ratio = accepted_quantity / max(quantity, 1e-6)
        net_value_per_ton = net_value / max(quantity, 1e-6)
        transport_cost_ratio = transport_cost / max(revenue, 1e-6)

        # Legacy sub-scores (kept for the "Score Breakdown" bars, still useful
        # as interpretable diagnostics alongside the ML prediction)
        economic_score = max(0, min(100, (net_value / max(quantity, 1)) / 20))
        distance_score = max(0, 100 - distance)
        capacity_score = min(100, accepted_quantity / max(quantity, 1) * 100)
        price_score = min(100, price / max_price * 100)

        if ml_model is not None:
            feature_row = pd.DataFrame([{
                "distance_km": distance,
                "price_per_ton": price,
                "capacity_tons": facility_capacity,
                "quantity_tons": quantity,
                "capacity_ratio": capacity_ratio,
                "net_value_per_ton": net_value_per_ton,
                "transport_cost_ratio": transport_cost_ratio,
                "trips": trips,
            }])[ML_FEATURES]
            final_score = float(np.clip(ml_model.predict(feature_row)[0], 0, 100))
        else:
            # Fallback if the model file isn't available
            final_score = (
                0.35 * economic_score
                + 0.30 * distance_score
                + 0.20 * capacity_score
                + 0.15 * price_score
            )

        rows.append({
            "name": row["name"],
            "city": row["city"],
            "type": row["type"],
            "waste": row["waste"],
            "capacity": facility_capacity,
            "accepted_quantity": accepted_quantity,
            "price": price,
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "distance": distance,
            "trips": trips,
            "revenue": revenue,
            "transport_cost": transport_cost,
            "net_value": net_value,
            "capacity_ratio": capacity_ratio,
            "net_value_per_ton": net_value_per_ton,
            "transport_cost_ratio": transport_cost_ratio,
            "economic_score": economic_score,
            "distance_score": distance_score,
            "capacity_score": capacity_score,
            "price_score": price_score,
            "score": final_score
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("score", ascending=False).reset_index(drop=True)
    return result

# ============================================================
# UTILIZATION METHODS (driven by utilization_methods.csv)
# ============================================================
CATEGORY_ICONS = {
    "Energy": "🔥",
    "Industrial": "🏭",
    "Construction": "🧱",
    "Biofuel": "⛽",
    "Agriculture": "🌱",
    "Materials": "🖤",
    "Chemical": "🧪",
}


def _potential_from_confidence(confidence_text):
    if not isinstance(confidence_text, str):
        return "Medium"
    text = confidence_text.strip().lower()
    if text.startswith("high"):
        return "High"
    if text.startswith("low"):
        return "Low"
    return "Medium"


def method_summary(waste_type):
    if methods_df.empty:
        return []

    subset = methods_df[
        methods_df["waste_type"].astype(str).str.lower() == waste_type.lower()
    ].copy()

    if subset.empty:
        return []

    # Prefer High/Medium confidence entries, keep top 4
    subset["_potential"] = subset["data_confidence"].apply(_potential_from_confidence)
    order = {"High": 0, "Medium": 1, "Low": 2}
    subset["_rank"] = subset["_potential"].map(order)
    subset = subset.sort_values("_rank").head(4)

    results = []
    for _, row in subset.iterrows():
        icon = CATEGORY_ICONS.get(row.get("category", ""), "♻️")
        name = row["utilization_method"]
        description = str(row.get("description", ""))
        if len(description) > 160:
            description = description[:157].rsplit(" ", 1)[0] + "..."
        potential = row["_potential"]
        results.append((icon, name, description, potential))
    return results

# ============================================================
# ENVIRONMENTAL SCORE
# ============================================================
def environmental_score(waste_type, utilization_type):
    if methods_df.empty:
        return 80
    match = methods_df[
        (methods_df["waste_type"].astype(str).str.lower() == waste_type.lower())
        & (methods_df["utilization_method"].astype(str).str.lower() == str(utilization_type).lower())
    ]
    if match.empty:
        return 80
    confidence = _potential_from_confidence(match.iloc[0].get("data_confidence", ""))
    if confidence == "High":
        return 88
    if confidence == "Medium":
        return 74
    return 60

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="font-size:30px; font-weight:800; color:#123d2a; margin-bottom:5px;">
            🌱 WasteWise AI
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("Agricultural waste utilization decision support")
    st.divider()
    st.subheader("Project Inputs")

    waste_type = st.selectbox(
        "♻️ Waste type",
        sorted(df["waste"].unique().tolist())
    )
    quantity = st.number_input(
        "⚖️ Quantity (tons)",
        min_value=0.5,
        max_value=500.0,
        value=10.0,
        step=0.5
    )
    city = st.selectbox("📍 Farmer location", list(LOCATIONS.keys()))

    st.divider()
    st.subheader("Prototype Settings")
    st.write("🚚 Truck capacity: **5 tons**")
    st.write("💰 Transport rate: **₹50/km/trip**")
    st.write("🏭 Facility prices: **Sample values**")
    st.warning("Replace sample facility data with verified local data before real-world deployment.")

    st.divider()
    st.subheader("🤖 ML Model")
    if ml_model is not None:
        st.success("Suitability model: **loaded** ✅")
        if model_metrics:
            st.caption(
                f"RandomForestRegressor • test R² = {model_metrics['test_r2']:.3f} • "
                f"test MAE = {model_metrics['test_mae']:.2f} pts"
            )
        st.caption(
            "Facility scores are predicted by a model trained on simulated "
            "historical transactions, not a fixed formula. See `train_model.py`."
        )
    else:
        st.error("Model file not found — using fallback formula. Run `python train_model.py`.")

# ============================================================
# INPUT SECTION
# ============================================================
st.markdown('<div class="section-title">🔍 Analyze Your Agricultural Waste</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="section-subtitle">
        Enter the waste details below to find the most suitable
        utilization opportunity and nearby facility.
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="input-card">', unsafe_allow_html=True)

input_col1, input_col2, input_col3, input_col4 = st.columns([1.4, 1, 1.2, 1])

with input_col1:
    st.write("**Waste Type**")
    st.info(f"♻️ {waste_type}")

with input_col2:
    st.write("**Quantity**")
    st.info(f"⚖️ {quantity:.1f} tons")

with input_col3:
    st.write("**Farmer Location**")
    st.info(f"📍 {city}")

with input_col4:
    st.write("**Ready?**")
    analyze = st.button("🔍 Analyze Waste", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# RUN ANALYSIS
# ============================================================
if analyze:
    st.session_state.analysis_results = calculate_recommendations(waste_type, quantity, city)
    st.session_state.analysis_waste_type = waste_type
    st.session_state.analysis_quantity = quantity
    st.session_state.analysis_city = city
    st.session_state.analysis_done = True

# ============================================================
# INITIAL SCREEN
# ============================================================
if not st.session_state.analysis_done:
    st.markdown(
        """
        <div class="info-card">
            <h3>🌱 Ready to Analyze</h3>
            <p>
                Select the agricultural waste type, quantity and
                farmer location from the sidebar, then click
                <b>Analyze Waste</b>.
            </p>
            <p style="color:#64748b;">
                WasteWise AI will compare compatible facilities
                using distance, capacity, price and estimated
                economic value.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="footer">
            🌱 <b>WasteWise AI</b>
            <br>
            Smart Agricultural Waste Utilization Decision Support System
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

# ============================================================
# LOAD STORED ANALYSIS
# ============================================================
results = st.session_state.analysis_results
waste_type = st.session_state.analysis_waste_type
quantity = st.session_state.analysis_quantity
city = st.session_state.analysis_city

if results.empty:
    st.error(f"No facilities were found for {waste_type}.")
    st.stop()

best = results.iloc[0]

# ============================================================
# ANALYSIS SUCCESS MESSAGE
# ============================================================
st.success(f"Analysis completed for {quantity:.1f} tons of {waste_type} from {city}.")
st.caption("💡 Change the inputs and click Analyze Waste again to generate a new recommendation.")

# ============================================================
# RECOMMENDATION
# ============================================================
st.markdown('<div class="section-title">🏆 Recommended Option</div>', unsafe_allow_html=True)
recommendation_col1, recommendation_col2 = st.columns([3.3, 1])

with recommendation_col1:
    st.markdown(
        f"""
        <div class="recommendation-card">
            <div class="recommendation-label">BEST MATCH</div>
            <div class="recommendation-title">{best["type"]}</div>
            <div class="recommendation-facility">🏭 {best["name"]} • 📍 {best["city"]}</div>
            <br>
            <div style="color:#52665c;">
                Recommended because it provides the best combined balance of
                economic value, distance, capacity and price.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with recommendation_col2:
    st.markdown(
        f"""
        <div class="score-box">
            <div class="score-label">OVERALL SCORE</div>
            <div class="score-number">{best["score"]:.0f}</div>
            <div class="score-label">out of 100</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# KEY METRICS
# ============================================================
metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">📍</div>
            <div class="metric-label">Distance</div>
            <div class="metric-value">{best["distance"]:.1f} km</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">🚚</div>
            <div class="metric-label">Transport Cost</div>
            <div class="metric-value">₹{best["transport_cost"]:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">💰</div>
            <div class="metric-label">Gross Value</div>
            <div class="metric-value">₹{best["revenue"]:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">📈</div>
            <div class="metric-label">Estimated Net Value</div>
            <div class="metric-value">₹{best["net_value"]:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# WHY WAS IT SELECTED?
# ============================================================
st.markdown('<div class="section-title">🤖 Why Was This Option Selected?</div>', unsafe_allow_html=True)
why_col1, why_col2 = st.columns([1.1, 1])

with why_col1:
    st.markdown('<div class="why-card">', unsafe_allow_html=True)
    reasons = []
    if best["distance"] <= results["distance"].median():
        reasons.append("📍 It is relatively close to the farmer, reducing transportation burden.")
    if best["price"] >= results["price"].median():
        reasons.append("💰 It offers a relatively strong value per ton.")
    if best["capacity"] >= quantity:
        reasons.append("🏭 Its listed capacity can handle the requested quantity.")
    else:
        reasons.append("⚠️ The facility has limited capacity, so only part of the requested quantity can be accepted.")

    for reason in reasons:
        st.markdown(f'<div class="reason">{reason}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with why_col2:
    st.markdown('<div class="why-card">', unsafe_allow_html=True)
    st.write("**Score Breakdown** (interpretable diagnostics)")
    score_data = pd.DataFrame({
        "Factor": ["Economic Value", "Distance", "Capacity", "Price"],
        "Score": [best["economic_score"], best["distance_score"], best["capacity_score"], best["price_score"]]
    })
    for _, score_row in score_data.iterrows():
        factor = score_row["Factor"]
        value = score_row["Score"]
        st.write(f"**{factor}** — {value:.0f}/100")
        st.progress(min(100, max(0, int(value))))
    st.markdown('</div>', unsafe_allow_html=True)

if ml_model is not None and feature_importance_df is not None:
    st.markdown("")
    st.markdown('<div class="why-card">', unsafe_allow_html=True)
    st.write("**🤖 What the ML model learned matters most**")
    st.caption(
        "Global feature importances from the trained RandomForestRegressor "
        "(computed once during training on simulated historical transactions, "
        "not recalculated per query)."
    )
    imp_display = feature_importance_df.copy()
    imp_display["importance_pct"] = (imp_display["importance"] * 100).round(1)
    imp_display = imp_display.set_index("feature")[["importance_pct"]]
    imp_display.columns = ["Importance (%)"]
    st.bar_chart(imp_display)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# UTILIZATION METHODS
# ============================================================
st.markdown('<div class="section-title">♻️ Possible Utilization Methods</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-subtitle">Research-backed ways to utilize <b>{waste_type}</b>.</div>',
    unsafe_allow_html=True
)

methods = method_summary(waste_type)

if not methods:
    st.info("No utilization method data available for this waste type.")
else:
    method_columns = st.columns(len(methods))
    for index, method in enumerate(methods):
        icon, name, description, potential = method
        potential_class = {
            "High": "potential-high",
            "Medium": "potential-medium",
            "Low": "potential-low",
        }.get(potential, "potential-medium")

        with method_columns[index]:
            st.markdown(
                f"""
                <div class="method-card">
                    <div style="font-size:32px;">{icon}</div>
                    <div class="method-name">{name}</div>
                    <div class="method-description">{description}</div>
                    <div class="{potential_class}">DATA CONFIDENCE: {potential.upper()}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ============================================================
# FACILITY RANKING
# ============================================================
st.markdown('<div class="section-title">🏭 Ranked Nearby Facilities</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="section-subtitle">
        Facilities are ranked using economic value, distance, capacity and price.
    </div>
    """,
    unsafe_allow_html=True
)

display_df = results[
    ["name", "city", "type", "distance", "capacity", "price", "transport_cost", "net_value", "score"]
].copy()
display_df.insert(0, "Rank", range(1, len(display_df) + 1))
display_df.columns = [
    "Rank", "Facility", "City", "Utilization", "Distance (km)",
    "Capacity (tons)", "Price (₹/ton)", "Transport (₹)", "Net Value (₹)", "Score"
]

st.dataframe(
    display_df.style.format({
        "Distance (km)": "{:.1f}",
        "Capacity (tons)": "{:.1f}",
        "Price (₹/ton)": "₹{:,.0f}",
        "Transport (₹)": "₹{:,.0f}",
        "Net Value (₹)": "₹{:,.0f}",
        "Score": "{:.0f}"
    }),
    use_container_width=True,
    hide_index=True
)

# ============================================================
# FACILITY NETWORK MAP
# ============================================================
st.markdown('<div class="section-title">🗺️ Facility Network Map</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-subtitle">Showing the farmer location in {city} and nearby compatible facilities.</div>',
    unsafe_allow_html=True
)

farmer_lat, farmer_lon = LOCATIONS[city]
m = folium.Map(location=[farmer_lat, farmer_lon], zoom_start=8, tiles="OpenStreetMap")

folium.Marker(
    [farmer_lat, farmer_lon],
    tooltip=f"👤 Farmer — {city}",
    popup=f"<b>Farmer Location</b><br>{city}",
    icon=folium.Icon(icon="user", prefix="fa", color="red")
).add_to(m)

for index, row in results.iterrows():
    is_best = index == 0
    popup_html = (
        f"<b>{row['name']}</b><br>"
        f"Type: {row['type']}<br>"
        f"City: {row['city']}<br>"
        f"Distance: {row['distance']:.1f} km<br>"
        f"Transport: ₹{row['transport_cost']:,.0f}<br>"
        f"Net value: ₹{row['net_value']:,.0f}<br>"
        f"Score: {row['score']:.0f}/100"
    )
    folium.Marker(
        [row["lat"], row["lon"]],
        tooltip=f"{'🏆 ' if is_best else '🏭 '}{row['name']} — Score {row['score']:.0f}",
        popup=popup_html,
        icon=folium.Icon(icon="star" if is_best else "industry", prefix="fa", color="green" if is_best else "blue")
    ).add_to(m)

folium.PolyLine(
    [[farmer_lat, farmer_lon], [best["lat"], best["lon"]]],
    tooltip=f"Recommended route: {best['distance']:.1f} km",
    weight=5
).add_to(m)

st_folium(m, width=None, height=500)

# ============================================================
# ECONOMIC IMPACT
# ============================================================
st.markdown('<div class="section-title">💰 Economic Impact</div>', unsafe_allow_html=True)
eco1, eco2, eco3 = st.columns(3)

with eco1:
    st.markdown(
        f"""
        <div class="info-card">
            <div style="font-size:28px;">💵</div>
            <div style="color:#64748b; font-size:13px; font-weight:600;">ESTIMATED GROSS VALUE</div>
            <div style="color:#123d2a; font-size:30px; font-weight:800;">₹{best["revenue"]:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with eco2:
    st.markdown(
        f"""
        <div class="info-card">
            <div style="font-size:28px;">🚚</div>
            <div style="color:#64748b; font-size:13px; font-weight:600;">TRANSPORT COST</div>
            <div style="color:#123d2a; font-size:30px; font-weight:800;">₹{best["transport_cost"]:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with eco3:
    st.markdown(
        f"""
        <div class="info-card">
            <div style="font-size:28px;">📈</div>
            <div style="color:#64748b; font-size:13px; font-weight:600;">ESTIMATED NET VALUE</div>
            <div style="color:#16803c; font-size:30px; font-weight:800;">₹{best["net_value"]:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# ENVIRONMENTAL IMPACT
# ============================================================
st.markdown('<div class="section-title">🌱 Environmental Impact</div>', unsafe_allow_html=True)
env_score = environmental_score(waste_type, best["type"])
env1, env2, env3 = st.columns(3)

with env1:
    st.markdown(
        f"""
        <div class="environment-card">
            <div style="font-size:28px;">♻️</div>
            <div style="color:#64748b; font-size:13px; font-weight:600;">WASTE DIVERTED</div>
            <div class="environment-number">{quantity:.1f} tons</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with env2:
    st.markdown(
        f"""
        <div class="environment-card">
            <div style="font-size:28px;">🌱</div>
            <div style="color:#64748b; font-size:13px; font-weight:600;">ESTIMATED SUSTAINABILITY SCORE</div>
            <div class="environment-number">{env_score}/100</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with env3:
    st.markdown(
        """
        <div class="environment-card">
            <div style="font-size:28px;">🔥</div>
            <div style="color:#64748b; font-size:13px; font-weight:600;">OPEN BURNING AVOIDED</div>
            <div class="environment-number">YES</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# DECISION SUMMARY
# ============================================================
st.markdown('<div class="section-title">📋 Decision Summary</div>', unsafe_allow_html=True)
summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.markdown(
        f"""
        <div class="info-card">
            <h3>🏆 Recommended Decision</h3>
            <p>For <b>{quantity:.1f} tons</b> of <b>{waste_type}</b> from <b>{city}</b>, the system recommends:</p>
            <h3 style="color:#2d8b5b;">{best["type"]}</h3>
            <p>Facility: <b>{best["name"]}</b></p>
            <p>Distance: <b>{best["distance"]:.1f} km</b></p>
            <p>Estimated net value: <b>₹{best["net_value"]:,.0f}</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

with summary_col2:
    st.markdown(
        """
        <div class="info-card">
            <h3>📊 Recommendation Basis</h3>
            <p>The system evaluates:</p>
            <p>📈 Economic value</p>
            <p>📍 Distance</p>
            <p>🏭 Facility capacity</p>
            <p>💰 Facility price</p>
            <p style="color:#64748b;">
                The final recommendation is generated using a weighted multi-factor scoring model.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        🌱 <b>WasteWise AI</b>
        <br>
        Smart Agricultural Waste Utilization Decision Support System
        <br><br>
        Prototype version • Facility data and pricing are sample values and should be verified before deployment.
    </div>
    """,
    unsafe_allow_html=True
)
