"""
Nassau Candy Distributor — Factory Optimization Dashboard
Best Model: Gradient Boosting Regressor
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy | Factory Optimizer",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Dark candy-wrapper theme */
[data-testid="stAppViewContainer"] {
    background: #0d0f14;
    color: #e8e8f0;
}
[data-testid="stSidebar"] {
    background: #13151d !important;
    border-right: 1px solid #1e2130;
}
[data-testid="stHeader"] { background: transparent; }

/* Metric cards */
.kpi-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #12141f 100%);
    border: 1px solid #2a2d45;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-card.purple::before { background: linear-gradient(90deg, #7c3aed, #a855f7); }
.kpi-card.teal::before   { background: linear-gradient(90deg, #0d9488, #34d399); }
.kpi-card.amber::before  { background: linear-gradient(90deg, #d97706, #fbbf24); }
.kpi-card.rose::before   { background: linear-gradient(90deg, #e11d48, #fb7185); }
.kpi-card.indigo::before { background: linear-gradient(90deg, #4f46e5, #818cf8); }
.kpi-card.cyan::before   { background: linear-gradient(90deg, #0891b2, #22d3ee); }

.kpi-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: #f1f5f9;
    font-family: 'DM Mono', monospace;
    line-height: 1;
}
.kpi-sub {
    font-size: 12px;
    color: #6b7280;
    margin-top: 4px;
}

/* Section headings */
.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #e2e8f0;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2130;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Alert boxes */
.alert-warning {
    background: rgba(217, 119, 6, 0.12);
    border: 1px solid rgba(217, 119, 6, 0.4);
    border-left: 4px solid #d97706;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #fcd34d;
    margin: 8px 0;
}
.alert-success {
    background: rgba(13, 148, 136, 0.12);
    border: 1px solid rgba(13, 148, 136, 0.4);
    border-left: 4px solid #0d9488;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #34d399;
    margin: 8px 0;
}
.alert-danger {
    background: rgba(225, 29, 72, 0.12);
    border: 1px solid rgba(225, 29, 72, 0.4);
    border-left: 4px solid #e11d48;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #fb7185;
    margin: 8px 0;
}

/* Recommendation row */
.rec-row {
    background: #1a1d2e;
    border: 1px solid #2a2d45;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.rec-rank {
    font-family: 'DM Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: #4f46e5;
    min-width: 40px;
}
.rec-product { font-weight: 600; color: #e2e8f0; font-size: 14px; }
.rec-route   { font-size: 12px; color: #6b7280; margin-top: 2px; }
.rec-badge   {
    background: rgba(13, 148, 136, 0.2);
    color: #34d399;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
}

/* Sidebar pill labels */
.sidebar-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7280;
}

/* Tab override */
[data-baseweb="tab-list"] {
    background: #13151d;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
[data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
}

/* Header banner */
.app-header {
    background: linear-gradient(135deg, #13151d 0%, #1a1d2e 50%, #0d0f14 100%);
    border: 1px solid #2a2d45;
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.app-title {
    font-size: 32px;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.app-subtitle { color: #6b7280; font-size: 14px; margin-top: 4px; }
.model-badge {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    padding: 8px 20px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
FACTORY_COORDS = {
    "Lot's O' Nuts":     (32.881893, -111.768036),
    "Wicked Choccy's":   (32.076176,  -81.088371),
    "Sugar Shack":       (48.11914,   -96.18115),
    "Secret Factory":    (41.446333,  -90.565487),
    "The Other Factory": (35.1175,    -89.971107),
}

REGION_COORDS = {
    "Interior": (39.5,  -100.0),
    "Atlantic":  (38.0,   -75.0),
    "Gulf":      (30.0,   -90.0),
    "Pacific":   (37.0,  -122.0),
}

FACTORY_MAP = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":          "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":     "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":         "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":  "Wicked Choccy's",
    "Laffy Taffy":                        "Sugar Shack",
    "SweeTARTS":                          "Sugar Shack",
    "Nerds":                              "Sugar Shack",
    "Fun Dip":                            "Sugar Shack",
    "Fizzy Lifting Drinks":               "Sugar Shack",
    "Everlasting Gobstopper":             "Secret Factory",
    "Hair Toffee":                        "The Other Factory",
    "Lickable Wallpaper":                 "Secret Factory",
    "Wonka Gum":                          "Secret Factory",
    "Kazookles":                          "The Other Factory",
}

SHIP_MODE_DAYS = {
    "Same Day": 1, "First Class": 3,
    "Second Class": 5, "Standard Class": 8,
}

DIVISIONS = ["Bulk", "Retail", "Wholesale"]
FEATURE_COLS = [
    "Ship Mode_enc", "Region_enc", "Division_enc",
    "Factory_enc", "Product Name_enc",
    "Shipping_Distance_km", "Base_Ship_Days",
    "Distance_x_ShipDays", "Profit_Margin",
    "Unit_Value", "Sales", "Units", "Cost",
    "Gross Profit", "Ship_x_Division"
]

FACTORY_COLORS = {
    "Lot's O' Nuts":     "#818cf8",
    "Wicked Choccy's":   "#34d399",
    "Sugar Shack":       "#fbbf24",
    "Secret Factory":    "#fb7185",
    "The Other Factory": "#22d3ee",
}


# ─── Utilities ────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# ─── Data & Model (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Building Gradient Boosting model…")
def build_model():
    np.random.seed(42)
    N = 4000

    products = list(FACTORY_MAP.keys())
    regions  = list(REGION_COORDS.keys())
    modes    = list(SHIP_MODE_DAYS.keys())

    df = pd.DataFrame({
        "Product Name": np.random.choice(products, N),
        "Region":       np.random.choice(regions,  N),
        "Ship Mode":    np.random.choice(modes,    N),
        "Division":     np.random.choice(DIVISIONS, N),
        "Sales":        np.random.uniform(50, 2000, N),
        "Units":        np.random.randint(1, 120, N),
        "Cost":         np.random.uniform(20, 1500, N),
    })
    df["Gross Profit"] = df["Sales"] - df["Cost"]
    df["Factory"] = df["Product Name"].map(FACTORY_MAP)

    df["Factory_Lat"] = df["Factory"].map(lambda f: FACTORY_COORDS[f][0])
    df["Factory_Lon"] = df["Factory"].map(lambda f: FACTORY_COORDS[f][1])
    df["Region_Lat"]  = df["Region"].map(lambda r: REGION_COORDS[r][0])
    df["Region_Lon"]  = df["Region"].map(lambda r: REGION_COORDS[r][1])

    df["Shipping_Distance_km"] = df.apply(
        lambda r: haversine_km(r["Factory_Lat"], r["Factory_Lon"],
                               r["Region_Lat"],  r["Region_Lon"]), axis=1)
    df["Base_Ship_Days"] = df["Ship Mode"].map(SHIP_MODE_DAYS)

    np.random.seed(0)
    combo_effects = {
        (reg, mode): np.random.uniform(-12, 12)
        for reg in regions for mode in modes
    }
    df["Route_Congestion_Effect"] = df.apply(
        lambda r: combo_effects[(r["Region"], r["Ship Mode"])], axis=1)

    df["Distance_x_ShipDays"] = df["Shipping_Distance_km"] * df["Base_Ship_Days"]
    df["Profit_Margin"]       = df["Gross Profit"] / (df["Sales"] + 1e-6)
    df["Unit_Value"]          = df["Sales"] / (df["Units"] + 1e-6)

    les = {}
    for col in ["Ship Mode", "Region", "Division", "Factory", "Product Name"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        les[col] = le

    df["Ship_x_Division"] = df["Ship Mode_enc"] * df["Division_enc"]

    np.random.seed(42)
    df["Lead_Time_Days"] = (
        df["Shipping_Distance_km"] * 0.008
        + df["Base_Ship_Days"]     * 10.0
        + df["Distance_x_ShipDays"] * 0.0005
        + df["Units"]              * 0.20
        + df["Route_Congestion_Effect"] * 2.0
        + np.random.normal(0, 8.5, N)
    ).round(1)

    Q1, Q3 = df["Lead_Time_Days"].quantile(0.01), df["Lead_Time_Days"].quantile(0.99)
    df_c = df[(df["Lead_Time_Days"] >= Q1) & (df["Lead_Time_Days"] <= Q3)].copy()
    df_c.reset_index(drop=True, inplace=True)

    X = df_c[FEATURE_COLS].values
    y = df_c["Lead_Time_Days"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)

    gb = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.10,
        max_depth=5, subsample=0.85, random_state=42
    )
    gb.fit(X_tr, y_tr)
    y_pred = gb.predict(X_te)

    r2   = r2_score(y_te, y_pred)
    rmse = mean_squared_error(y_te, y_pred) ** 0.5
    mae  = mean_absolute_error(y_te, y_pred)

    return gb, les, scaler, df_c, r2, rmse, mae, X_te, y_te, y_pred


def predict_lt(gb, les, df_c, product, factory, region, ship_mode, priority_w):
    dist = haversine_km(
        FACTORY_COORDS[factory][0], FACTORY_COORDS[factory][1],
        REGION_COORDS[region][0],   REGION_COORDS[region][1],
    )
    base   = SHIP_MODE_DAYS[ship_mode]
    dx_sd  = dist * base
    pdata  = df_c[df_c["Product Name"] == product]
    if len(pdata) == 0:
        pdata  = df_c
    avg_s  = pdata["Sales"].mean()
    avg_u  = pdata["Units"].mean()
    avg_c  = pdata["Cost"].mean()
    avg_p  = pdata["Gross Profit"].mean()
    margin = avg_p / (avg_s + 1e-6)
    uval   = avg_s / (avg_u + 1e-6)
    div    = df_c[df_c["Product Name"] == product]["Division"].iloc[0] \
             if len(df_c[df_c["Product Name"] == product]) > 0 else "Retail"

    try:
        ship_enc = les["Ship Mode"].transform([ship_mode])[0]
        reg_enc  = les["Region"].transform([region])[0]
        div_enc  = les["Division"].transform([div])[0]
        fac_enc  = les["Factory"].transform([factory])[0]
        prd_enc  = les["Product Name"].transform([product])[0]
    except Exception:
        return None

    row = np.array([[
        ship_enc, reg_enc, div_enc, fac_enc, prd_enc,
        dist, base, dx_sd, margin, uval,
        avg_s, avg_u, avg_c, avg_p, ship_enc * div_enc
    ]])
    raw = gb.predict(row)[0]

    # Priority slider: 0 = speed, 1 = profit
    speed_factor = (1 - priority_w) * (dist / 5000) * 3
    profit_bonus = priority_w * (avg_p / 200)
    adjusted = raw - speed_factor + profit_bonus
    return max(adjusted, 1.0)


# ─── Load ─────────────────────────────────────────────────────────────────────
gb_model, label_encoders, scaler, df_clean, r2, rmse, mae, X_te, y_te, y_pred = build_model()

ALL_FACTORIES = list(FACTORY_COORDS.keys())
ALL_PRODUCTS  = list(FACTORY_MAP.keys())
ALL_REGIONS   = list(REGION_COORDS.keys())
ALL_MODES     = list(SHIP_MODE_DAYS.keys())


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 24px 0;">
        <div style="font-size:40px;">🍬</div>
        <div style="font-size:16px; font-weight:700; color:#e2e8f0; margin-top:6px;">Nassau Candy</div>
        <div style="font-size:11px; color:#6b7280; letter-spacing:0.1em; text-transform:uppercase;">Factory Optimizer</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-label">📦 Product Filter</p>', unsafe_allow_html=True)
    selected_product = st.selectbox("", ALL_PRODUCTS, label_visibility="collapsed")

    st.markdown('<p class="sidebar-label" style="margin-top:16px">🌎 Region</p>', unsafe_allow_html=True)
    selected_region = st.selectbox("", ALL_REGIONS, label_visibility="collapsed", key="region")

    st.markdown('<p class="sidebar-label" style="margin-top:16px">🚢 Ship Mode</p>', unsafe_allow_html=True)
    selected_mode = st.selectbox("", ALL_MODES, label_visibility="collapsed", key="mode")

    st.markdown('<p class="sidebar-label" style="margin-top:16px">⚡ Optimization Priority</p>', unsafe_allow_html=True)
    priority = st.slider(
        "", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        label_visibility="collapsed",
        help="0 = Minimize lead time (speed), 1 = Maximize profit efficiency"
    )
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;font-size:11px;color:#6b7280;">'
        f'<span>⚡ Speed</span><span>💰 Profit</span></div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="background:#1a1d2e; border-radius:10px; padding:14px; border:1px solid #2a2d45;">
        <div style="font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:10px;">Model Info</div>
        <div style="font-size:13px; color:#818cf8; font-weight:600;">Gradient Boosting</div>
        <div style="font-size:11px; color:#6b7280; margin-top:4px;">Best performing model</div>
        <div style="margin-top:12px; display:flex; justify-content:space-between;">
            <span style="font-size:12px; color:#34d399;">R² {r2*100:.1f}%</span>
            <span style="font-size:12px; color:#fbbf24;">RMSE {rmse:.1f}d</span>
            <span style="font-size:12px; color:#fb7185;">MAE {mae:.1f}d</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div>
    <h1 class="app-title">🍬 Nassau Candy Optimizer</h1>
    <p class="app-subtitle">Factory assignment intelligence · Lead-time reduction · Profit-aware routing</p>
  </div>
  <div class="model-badge">✦ Gradient Boosting · Best Model</div>
</div>
""", unsafe_allow_html=True)


# ─── KPI Row ──────────────────────────────────────────────────────────────────
current_factory = FACTORY_MAP[selected_product]
current_lt = predict_lt(gb_model, label_encoders, df_clean,
                        selected_product, current_factory,
                        selected_region, selected_mode, priority)

# Find best alternative
best_alt, best_lt, best_imp = None, 9999, 0
for alt in ALL_FACTORIES:
    if alt == current_factory:
        continue
    alt_lt = predict_lt(gb_model, label_encoders, df_clean,
                        selected_product, alt, selected_region, selected_mode, priority)
    if alt_lt and alt_lt < best_lt:
        best_lt, best_alt = alt_lt, alt

if best_lt and current_lt:
    best_imp = ((current_lt - best_lt) / current_lt) * 100

pdata = df_clean[df_clean["Product Name"] == selected_product]
avg_profit = pdata["Gross Profit"].mean() if len(pdata) else 45.0
profit_impact = avg_profit * (best_imp / 100) * 0.5 if best_imp > 0 else 0

k1, k2, k3, k4, k5, k6 = st.columns(6)
cards = [
    (k1, "purple", "CURRENT LEAD TIME", f"{current_lt:.1f}d", f"{current_factory[:18]}"),
    (k2, "teal",   "BEST FACTORY",      best_alt[:16] if best_alt else "—", f"Recommended"),
    (k3, "amber",  "LEAD TIME SAVING",  f"{best_imp:.1f}%", f"{(current_lt - best_lt):.1f} days" if best_lt else ""),
    (k4, "rose",   "PROFIT IMPACT",     f"${profit_impact:.0f}", "per order avg"),
    (k5, "indigo", "MODEL R² SCORE",    f"{r2*100:.1f}%", "Gradient Boosting"),
    (k6, "cyan",   "PRODUCTS COVERED",  str(len(ALL_PRODUCTS)), "across 5 factories"),
]
for col, color, label, value, sub in cards:
    with col:
        st.markdown(f"""
        <div class="kpi-card {color}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Main Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏭 Factory Simulator",
    "🔄 What-If Scenarios",
    "🏆 Recommendations",
    "⚠️ Risk & Impact",
])


# ══════════════════════════════════════════════════════
# TAB 1 — Factory Optimization Simulator
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">🏭 Factory Performance Simulator</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])

    with col_a:
        # Predict across all factories for selected product / region / mode
        factory_results = {}
        for fac in ALL_FACTORIES:
            lt = predict_lt(gb_model, label_encoders, df_clean,
                            selected_product, fac, selected_region, selected_mode, priority)
            factory_results[fac] = lt if lt else 0

        fac_df = pd.DataFrame([
            {"Factory": f, "Lead Time (days)": lt,
             "Distance (km)": haversine_km(
                 FACTORY_COORDS[f][0], FACTORY_COORDS[f][1],
                 REGION_COORDS[selected_region][0], REGION_COORDS[selected_region][1]),
             "Current": "★ Current" if f == current_factory else "Alternative"}
            for f, lt in factory_results.items()
        ]).sort_values("Lead Time (days)")

        colors = [FACTORY_COLORS.get(f, "#818cf8") for f in fac_df["Factory"]]

        fig = go.Figure()
        for i, row in fac_df.iterrows():
            fig.add_trace(go.Bar(
                y=[row["Factory"]],
                x=[row["Lead Time (days)"]],
                orientation='h',
                name=row["Factory"],
                marker_color=FACTORY_COLORS.get(row["Factory"], "#818cf8"),
                marker_line_color='rgba(255,255,255,0.1)',
                marker_line_width=1,
                text=f"  {row['Lead Time (days)']:.1f}d",
                textposition='outside',
                textfont=dict(color='#e2e8f0', size=12, family='DM Mono'),
                hovertemplate=(
                    f"<b>{row['Factory']}</b><br>"
                    f"Lead Time: {row['Lead Time (days)']:.1f} days<br>"
                    f"Distance: {row['Distance (km)']:.0f} km<br>"
                    f"<extra></extra>"
                ),
                showlegend=False,
            ))

        fig.update_layout(
            title=dict(
                text=f"Predicted Lead Time by Factory<br>"
                     f"<sup>{selected_product} → {selected_region} · {selected_mode}</sup>",
                font=dict(color='#e2e8f0', size=15),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,29,46,0.6)',
            xaxis=dict(
                title="Lead Time (days)",
                color='#6b7280', gridcolor='#1e2130',
                zeroline=False,
            ),
            yaxis=dict(color='#e2e8f0', gridcolor='rgba(0,0,0,0)'),
            height=350,
            margin=dict(l=10, r=60, t=70, b=10),
            font=dict(family='Space Grotesk'),
        )
        # Highlight current factory
        current_idx = fac_df[fac_df["Factory"] == current_factory].index
        if len(current_idx):
            fig.add_vline(
                x=factory_results[current_factory],
                line_dash="dash", line_color="rgba(251,191,36,0.5)",
                annotation_text="Current", annotation_font_color="#fbbf24",
            )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Factory Details**")
        for fac in fac_df["Factory"].tolist():
            lt_val = factory_results[fac]
            is_cur = fac == current_factory
            is_best = fac == fac_df["Factory"].iloc[0] and not is_cur
            icon = "★" if is_cur else ("🏆" if is_best else "○")
            color = "#fbbf24" if is_cur else ("#34d399" if is_best else "#6b7280")
            st.markdown(f"""
            <div style="padding:10px 14px; margin-bottom:8px;
                        background:#1a1d2e; border-radius:10px;
                        border:1px solid {'#fbbf24' if is_cur else '#2a2d45'};">
                <div style="font-size:12px; color:{color}; font-weight:600;">{icon} {fac}</div>
                <div style="font-size:20px; color:#e2e8f0; font-family:'DM Mono',monospace; font-weight:700;">
                    {lt_val:.1f}<span style="font-size:12px; color:#6b7280;"> days</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Factory map
    st.markdown("---")
    st.markdown("**📍 Factory & Region Locations**")

    map_data = []
    for name, (lat, lon) in FACTORY_COORDS.items():
        map_data.append({
            "name": name, "lat": lat, "lon": lon,
            "type": "Factory", "lt": factory_results.get(name, 0),
            "color": FACTORY_COLORS.get(name, "#818cf8"),
        })
    for name, (lat, lon) in REGION_COORDS.items():
        map_data.append({
            "name": name, "lat": lat, "lon": lon,
            "type": "Region", "lt": 0, "color": "#6b7280",
        })
    map_df = pd.DataFrame(map_data)

    fig_map = px.scatter_mapbox(
        map_df, lat="lat", lon="lon", hover_name="name",
        color="type",
        color_discrete_map={"Factory": "#818cf8", "Region": "#34d399"},
        size=[20 if t == "Factory" else 12 for t in map_df["type"]],
        size_max=18,
        zoom=3.5, center={"lat": 38, "lon": -97},
        mapbox_style="carto-darkmatter",
        height=380,
    )
    fig_map.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color='#e2e8f0')),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig_map, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 2 — What-If Scenario Analysis
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🔄 What-If Scenario Analysis</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("**Compare across all regions**")
        region_scenario = []
        for reg in ALL_REGIONS:
            cur_lt = predict_lt(gb_model, label_encoders, df_clean,
                                selected_product, current_factory,
                                reg, selected_mode, priority)
            rec_lt_val = predict_lt(gb_model, label_encoders, df_clean,
                                    selected_product, best_alt or current_factory,
                                    reg, selected_mode, priority)
            if cur_lt and rec_lt_val:
                region_scenario.append({
                    "Region": reg,
                    "Current Factory": round(cur_lt, 1),
                    "Recommended Factory": round(rec_lt_val, 1),
                    "Saving (days)": round(cur_lt - rec_lt_val, 1),
                })

        rs_df = pd.DataFrame(region_scenario)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=rs_df["Region"], y=rs_df["Current Factory"],
            name=current_factory[:20],
            marker_color="#fb7185",
            text=rs_df["Current Factory"],
            textposition="outside",
            textfont=dict(size=11, family="DM Mono", color="#e2e8f0"),
        ))
        fig2.add_trace(go.Bar(
            x=rs_df["Region"], y=rs_df["Recommended Factory"],
            name=f"{best_alt[:20] if best_alt else '—'} (Rec.)",
            marker_color="#34d399",
            text=rs_df["Recommended Factory"],
            textposition="outside",
            textfont=dict(size=11, family="DM Mono", color="#e2e8f0"),
        ))
        fig2.update_layout(
            barmode="group",
            title=dict(text="Lead Time: Current vs Recommended by Region",
                       font=dict(color="#e2e8f0")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,29,46,0.6)",
            xaxis=dict(color="#6b7280", gridcolor="#1e2130"),
            yaxis=dict(title="Lead Time (days)", color="#6b7280", gridcolor="#1e2130"),
            legend=dict(font=dict(color="#e2e8f0"), bgcolor="rgba(0,0,0,0)"),
            height=340,
            font=dict(family="Space Grotesk"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        st.markdown("**Compare across ship modes**")
        mode_scenario = []
        for mode in ALL_MODES:
            cur_lt_m = predict_lt(gb_model, label_encoders, df_clean,
                                  selected_product, current_factory,
                                  selected_region, mode, priority)
            rec_lt_m = predict_lt(gb_model, label_encoders, df_clean,
                                  selected_product, best_alt or current_factory,
                                  selected_region, mode, priority)
            if cur_lt_m and rec_lt_m:
                mode_scenario.append({
                    "Mode": mode,
                    "Current": round(cur_lt_m, 1),
                    "Recommended": round(rec_lt_m, 1),
                    "Delta": round(cur_lt_m - rec_lt_m, 1),
                })

        ms_df = pd.DataFrame(mode_scenario)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=ms_df["Mode"], y=ms_df["Current"],
            mode="lines+markers+text",
            name="Current Factory",
            line=dict(color="#fb7185", width=2.5),
            marker=dict(size=9, symbol="circle"),
            text=ms_df["Current"], textposition="top center",
            textfont=dict(size=10, family="DM Mono", color="#fb7185"),
        ))
        fig3.add_trace(go.Scatter(
            x=ms_df["Mode"], y=ms_df["Recommended"],
            mode="lines+markers+text",
            name="Recommended",
            line=dict(color="#34d399", width=2.5, dash="dash"),
            marker=dict(size=9, symbol="diamond"),
            text=ms_df["Recommended"], textposition="bottom center",
            textfont=dict(size=10, family="DM Mono", color="#34d399"),
        ))
        fig3.update_layout(
            title=dict(text="Lead Time by Ship Mode",
                       font=dict(color="#e2e8f0")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,29,46,0.6)",
            xaxis=dict(color="#6b7280", gridcolor="#1e2130"),
            yaxis=dict(title="Lead Time (days)", color="#6b7280", gridcolor="#1e2130"),
            legend=dict(font=dict(color="#e2e8f0"), bgcolor="rgba(0,0,0,0)"),
            height=340,
            font=dict(family="Space Grotesk"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Savings heatmap
    st.markdown("---")
    st.markdown("**🌡️ Improvement Heatmap: Factory × Region**")

    heat_data = []
    for fac in ALL_FACTORIES:
        row_data = {"Factory": fac}
        for reg in ALL_REGIONS:
            cur_lt_h = predict_lt(gb_model, label_encoders, df_clean,
                                  selected_product, current_factory,
                                  reg, selected_mode, priority)
            alt_lt_h = predict_lt(gb_model, label_encoders, df_clean,
                                  selected_product, fac, reg, selected_mode, priority)
            if cur_lt_h and alt_lt_h and cur_lt_h > 0:
                row_data[reg] = round(((cur_lt_h - alt_lt_h) / cur_lt_h) * 100, 1)
            else:
                row_data[reg] = 0
        heat_data.append(row_data)

    heat_df = pd.DataFrame(heat_data).set_index("Factory")

    fig_heat = px.imshow(
        heat_df,
        color_continuous_scale=[[0, "#e11d48"], [0.5, "#1a1d2e"], [1, "#059669"]],
        color_continuous_midpoint=0,
        text_auto=".1f",
        aspect="auto",
        title="Lead Time Improvement (%) vs Current Factory",
    )
    fig_heat.update_traces(textfont_size=12, textfont_family="DM Mono")
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Space Grotesk"),
        title_font=dict(color="#e2e8f0"),
        coloraxis_colorbar=dict(tickfont=dict(color="#6b7280")),
        xaxis=dict(tickfont=dict(color="#e2e8f0")),
        yaxis=dict(tickfont=dict(color="#e2e8f0")),
        height=280,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 3 — Recommendation Dashboard
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">🏆 Ranked Reassignment Recommendations</div>', unsafe_allow_html=True)

    # Build recommendations for all products
    all_recs = []
    for product in ALL_PRODUCTS:
        cur_fac = FACTORY_MAP[product]
        best_fac_r, best_lt_r, best_imp_r = None, 9999, -999
        for alt_fac in ALL_FACTORIES:
            if alt_fac == cur_fac:
                continue
            # Average across all regions & modes
            lts = []
            cur_lts = []
            for reg in ALL_REGIONS:
                for mode in ALL_MODES:
                    a = predict_lt(gb_model, label_encoders, df_clean,
                                   product, alt_fac, reg, mode, priority)
                    c = predict_lt(gb_model, label_encoders, df_clean,
                                   product, cur_fac, reg, mode, priority)
                    if a and c:
                        lts.append(a)
                        cur_lts.append(c)
            if lts:
                avg_alt = np.mean(lts)
                avg_cur = np.mean(cur_lts)
                imp_pct = ((avg_cur - avg_alt) / avg_cur) * 100
                if imp_pct > best_imp_r:
                    best_imp_r = imp_pct
                    best_lt_r  = avg_alt
                    best_fac_r = alt_fac
                    cur_lt_r   = avg_cur

        if best_fac_r and best_imp_r > 0:
            pd_data = df_clean[df_clean["Product Name"] == product]
            avg_p   = pd_data["Gross Profit"].mean() if len(pd_data) else 30
            conf    = min(0.9, max(0.3, best_imp_r / 30 * 0.6 + 0.3))
            all_recs.append({
                "Product": product,
                "Current Factory": cur_fac,
                "Recommended":     best_fac_r,
                "Improvement (%)": round(best_imp_r, 1),
                "Days Saved":      round(cur_lt_r - best_lt_r, 1),
                "Confidence":      round(conf, 2),
                "Avg Profit ($)":  round(avg_p, 1),
            })

    all_recs_df = pd.DataFrame(all_recs).sort_values("Improvement (%)", ascending=False).reset_index(drop=True)

    # Rank display
    st.markdown(f"**{len(all_recs_df)} reassignment opportunities found**")
    for i, row in all_recs_df.iterrows():
        conf_pct = int(row["Confidence"] * 100)
        conf_color = "#34d399" if conf_pct >= 60 else "#fbbf24"
        st.markdown(f"""
        <div class="rec-row">
            <div style="display:flex; align-items:center; gap:16px; flex:1;">
                <div class="rec-rank">#{i+1}</div>
                <div>
                    <div class="rec-product">{row["Product"]}</div>
                    <div class="rec-route">
                        <span style="color:#fb7185;">{row["Current Factory"]}</span>
                        &nbsp;→&nbsp;
                        <span style="color:#34d399;">{row["Recommended"]}</span>
                    </div>
                </div>
            </div>
            <div style="display:flex; gap:12px; align-items:center;">
                <div style="text-align:center;">
                    <div style="font-size:10px; color:#6b7280; text-transform:uppercase;">Days Saved</div>
                    <div style="font-family:'DM Mono'; color:#fbbf24; font-weight:700;">{row["Days Saved"]:.1f}d</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:10px; color:#6b7280; text-transform:uppercase;">Confidence</div>
                    <div style="font-family:'DM Mono'; color:{conf_color}; font-weight:700;">{conf_pct}%</div>
                </div>
                <div class="rec-badge">+{row["Improvement (%)"]:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        fig_imp = px.bar(
            all_recs_df.head(10).sort_values("Improvement (%)"),
            x="Improvement (%)", y="Product", orientation="h",
            color="Improvement (%)",
            color_continuous_scale=[[0, "#4f46e5"], [1, "#34d399"]],
            title="Top 10 Improvement Opportunities",
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,29,46,0.6)",
            font=dict(color="#e2e8f0", family="Space Grotesk"),
            title_font=dict(color="#e2e8f0"),
            coloraxis_showscale=False,
            height=380,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis=dict(color="#6b7280", gridcolor="#1e2130"),
            yaxis=dict(color="#e2e8f0"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_chart2:
        fig_conf = px.scatter(
            all_recs_df,
            x="Improvement (%)", y="Confidence",
            size="Days Saved", color="Recommended",
            hover_name="Product",
            title="Improvement vs Confidence",
            color_discrete_sequence=["#818cf8", "#34d399", "#fbbf24", "#fb7185", "#22d3ee"],
        )
        fig_conf.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,29,46,0.6)",
            font=dict(color="#e2e8f0", family="Space Grotesk"),
            title_font=dict(color="#e2e8f0"),
            legend=dict(font=dict(color="#e2e8f0"), bgcolor="rgba(0,0,0,0)"),
            height=380,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis=dict(color="#6b7280", gridcolor="#1e2130"),
            yaxis=dict(color="#6b7280", gridcolor="#1e2130"),
        )
        st.plotly_chart(fig_conf, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 4 — Risk & Impact Panel
# ══════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">⚠️ Risk & Impact Analysis</div>', unsafe_allow_html=True)

    col_risk1, col_risk2 = st.columns([1, 1])

    with col_risk1:
        st.markdown("**📊 Profit Impact Analysis**")

        profit_data = []
        for product in ALL_PRODUCTS:
            pd_data  = df_clean[df_clean["Product Name"] == product]
            avg_p    = pd_data["Gross Profit"].mean() if len(pd_data) else 30
            cur_fac  = FACTORY_MAP[product]
            best_fac_p, best_imp_p = None, 0
            for alt_fac in ALL_FACTORIES:
                if alt_fac == cur_fac:
                    continue
                cur_l = predict_lt(gb_model, label_encoders, df_clean,
                                   product, cur_fac, "Atlantic", "Standard Class", priority)
                alt_l = predict_lt(gb_model, label_encoders, df_clean,
                                   product, alt_fac, "Atlantic", "Standard Class", priority)
                if cur_l and alt_l and cur_l > 0:
                    imp = ((cur_l - alt_l) / cur_l) * 100
                    if imp > best_imp_p:
                        best_imp_p = imp
                        best_fac_p = alt_fac

            profit_uplift = avg_p * (best_imp_p / 100) * 0.4
            profit_data.append({
                "Product":        product[:25],
                "Avg Profit ($)": round(avg_p, 1),
                "Profit Uplift":  round(profit_uplift, 1),
                "Improvement %":  round(best_imp_p, 1),
                "Best Factory":   best_fac_p or cur_fac,
            })

        profit_df = pd.DataFrame(profit_data).sort_values("Profit Uplift", ascending=False)

        fig_profit = px.bar(
            profit_df,
            x="Profit Uplift", y="Product", orientation="h",
            color="Improvement %",
            color_continuous_scale=[[0, "#1e2130"], [0.5, "#4f46e5"], [1, "#34d399"]],
            title="Expected Profit Uplift per Product ($)",
            hover_data=["Avg Profit ($)", "Best Factory"],
        )
        fig_profit.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,29,46,0.6)",
            font=dict(color="#e2e8f0", family="Space Grotesk"),
            title_font=dict(color="#e2e8f0"),
            height=480,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis=dict(title="Estimated Profit Uplift ($)", color="#6b7280", gridcolor="#1e2130"),
            yaxis=dict(color="#e2e8f0"),
        )
        st.plotly_chart(fig_profit, use_container_width=True)

    with col_risk2:
        st.markdown("**🚦 Risk Classification**")

        risk_alerts = []
        for product in ALL_PRODUCTS:
            cur_fac = FACTORY_MAP[product]
            pd_data = df_clean[df_clean["Product Name"] == product]
            avg_p   = pd_data["Gross Profit"].mean() if len(pd_data) else 30

            cur_l = predict_lt(gb_model, label_encoders, df_clean,
                               product, cur_fac, selected_region, selected_mode, priority)
            if not cur_l:
                continue

            best_lt_risk, best_fac_risk, best_imp_risk = 9999, None, 0
            for alt in ALL_FACTORIES:
                if alt == cur_fac:
                    continue
                alt_l = predict_lt(gb_model, label_encoders, df_clean,
                                   product, alt, selected_region, selected_mode, priority)
                if alt_l and alt_l < best_lt_risk:
                    best_lt_risk = alt_l
                    imp_r = ((cur_l - alt_l) / cur_l) * 100
                    if imp_r > best_imp_risk:
                        best_imp_risk = imp_r
                        best_fac_risk = alt

            # Risk = high lead time & low profit
            risk_score = (cur_l / 100) + (1 / (avg_p + 1))
            if cur_l > 80:
                level, cls = "HIGH", "alert-danger"
            elif cur_l > 50:
                level, cls = "MEDIUM", "alert-warning"
            else:
                level, cls = "LOW", "alert-success"

            risk_alerts.append((risk_score, level, cls, product, cur_fac, cur_l,
                                 best_fac_risk, best_imp_risk))

        risk_alerts.sort(reverse=True)

        st.markdown("**Products by Risk Level**")
        for _, level, cls, prod, cur_f, cur_l_v, best_f, best_i in risk_alerts[:8]:
            st.markdown(f"""
            <div class="{cls}" style="font-size:12px;">
                <strong>[{level}]</strong> {prod[:30]}<br>
                <span style="opacity:0.8;">
                    Factory: {cur_f} · Lead time: {cur_l_v:.1f}d
                    {f' · Recommend → {best_f} (+{best_i:.0f}%)' if best_f and best_i > 0 else ''}
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**🎯 Model Actual vs Predicted**")

        sample_idx = np.random.choice(len(y_te), min(300, len(y_te)), replace=False)
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=y_te[sample_idx], y=y_pred[sample_idx],
            mode="markers",
            marker=dict(
                color=np.abs(y_te[sample_idx] - y_pred[sample_idx]),
                colorscale=[[0, "#34d399"], [0.5, "#fbbf24"], [1, "#fb7185"]],
                size=5, opacity=0.7,
                colorbar=dict(title="Error", tickfont=dict(color="#6b7280")),
            ),
            hovertemplate="Actual: %{x:.1f}d<br>Predicted: %{y:.1f}d<extra></extra>",
            showlegend=False,
        ))
        lim = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
        fig_avp.add_trace(go.Scatter(
            x=lim, y=lim,
            mode="lines",
            line=dict(color="#4f46e5", dash="dash", width=2),
            name="Perfect fit",
        ))
        fig_avp.update_layout(
            title=dict(text=f"Actual vs Predicted  ·  R² = {r2*100:.1f}%",
                       font=dict(color="#e2e8f0")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,29,46,0.6)",
            xaxis=dict(title="Actual (days)", color="#6b7280", gridcolor="#1e2130"),
            yaxis=dict(title="Predicted (days)", color="#6b7280", gridcolor="#1e2130"),
            legend=dict(font=dict(color="#e2e8f0"), bgcolor="rgba(0,0,0,0)"),
            height=320,
            font=dict(family="Space Grotesk"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_avp, use_container_width=True)

    # Feature importance
    st.markdown("---")
    st.markdown('<div class="section-title">🔬 Feature Importance — Gradient Boosting</div>',
                unsafe_allow_html=True)

    feat_imp = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": gb_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    fig_fi = px.bar(
        feat_imp,
        x="Feature", y="Importance",
        color="Importance",
        color_continuous_scale=[[0, "#1e2130"], [0.5, "#4f46e5"], [1, "#818cf8"]],
        title="Feature Importance Scores",
    )
    fig_fi.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,46,0.6)",
        font=dict(color="#e2e8f0", family="Space Grotesk"),
        title_font=dict(color="#e2e8f0"),
        xaxis=dict(color="#6b7280", gridcolor="#1e2130", tickangle=30),
        yaxis=dict(color="#6b7280", gridcolor="#1e2130"),
        coloraxis_showscale=False,
        height=300,
        margin=dict(l=10, r=10, t=50, b=60),
    )
    st.plotly_chart(fig_fi, use_container_width=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:32px 0 16px 0; color:#2a2d45; font-size:12px;">
    Nassau Candy Distributor · Factory Optimization Dashboard ·
    Powered by Gradient Boosting Regressor
</div>
""", unsafe_allow_html=True)