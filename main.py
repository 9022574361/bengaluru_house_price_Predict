import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# --- PREMIUM CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── ROOT TOKENS ─────────────────────────────────────────── */
:root {
    --gold:        #C9A84C;
    --gold-light:  #E8C97A;
    --gold-dim:    rgba(201,168,76,0.18);
    --dark-bg:     #0E0F13;
    --card-bg:     #16181F;
    --card-border: rgba(201,168,76,0.22);
    --surface:     #1E2029;
    --text-primary:#F0EBE0;
    --text-muted:  #8A8678;
    --danger:      #E05A5A;
    --radius-lg:   16px;
    --radius-md:   10px;
    --shadow-gold: 0 0 40px rgba(201,168,76,0.12);
}

/* ── GLOBAL RESET ─────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
    background-color: var(--dark-bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 780px !important;
}

/* ── HIDE STREAMLIT CHROME ────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── HERO BANNER ──────────────────────────────────────────── */
.hero-wrap {
    background: linear-gradient(135deg, #1a1b22 0%, #0e0f13 60%, #1a1408 100%);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-gold);
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(201,168,76,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(201,168,76,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.9rem, 4vw, 2.8rem);
    font-weight: 700;
    line-height: 1.2;
    color: var(--text-primary);
    margin-bottom: 0.9rem;
}
.hero-title span { color: var(--gold); }
.hero-subtitle {
    font-size: 0.9rem;
    color: var(--text-muted);
    max-width: 480px;
    line-height: 1.65;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--gold-dim);
    border: 1px solid var(--card-border);
    border-radius: 50px;
    padding: 5px 14px;
    font-size: 0.75rem;
    color: var(--gold-light);
    font-weight: 500;
    margin-top: 1.4rem;
}
.hero-badge::before { content: '◈'; font-size: 0.7rem; }

/* ── SECTION LABEL ────────────────────────────────────────── */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 1rem;
    margin-top: 2rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--gold-dim), transparent);
}

/* ── INPUT CARD ───────────────────────────────────────────── */
.input-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: border-color 0.25s;
}
.input-card:hover { border-color: rgba(201,168,76,0.4); }

/* ── STREAMLIT WIDGET OVERRIDES ───────────────────────────── */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: var(--surface) !important;
    border: 1.5px solid rgba(201,168,76,0.25) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 0.9rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(201,168,76,0.15) !important;
    outline: none !important;
}

/* Slider track */
div[data-testid="stSlider"] .st-bu { background: var(--gold) !important; }
div[data-testid="stSlider"] .st-bv { background: var(--surface) !important; }
div[data-testid="stSlider"] [role="slider"] {
    background: var(--gold) !important;
    border: 2px solid var(--dark-bg) !important;
    width: 18px !important; height: 18px !important;
    box-shadow: 0 0 10px rgba(201,168,76,0.5) !important;
}
div[data-testid="stSlider"] p {
    color: var(--gold-light) !important;
    font-weight: 600 !important;
}

/* Selectbox */
div[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border: 1.5px solid rgba(201,168,76,0.25) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
}
div[data-testid="stSelectbox"] span { color: var(--text-primary) !important; }

/* Widget labels */
div[data-testid="stWidgetLabel"] p,
label, .st-emotion-cache-1inwz65 {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

/* Number input arrows */
div[data-testid="stNumberInput"] button {
    background: var(--surface) !important;
    border-color: rgba(201,168,76,0.2) !important;
    color: var(--gold) !important;
}

/* ── PREDICT BUTTON ───────────────────────────────────────── */
div[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #C9A84C 0%, #A0782A 100%) !important;
    color: #0E0F13 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    height: 3.2rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(201,168,76,0.3) !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #E8C97A 0%, #C9A84C 100%) !important;
    box-shadow: 0 6px 28px rgba(201,168,76,0.45) !important;
    transform: translateY(-2px) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── METRICS ROW ──────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid rgba(201,168,76,0.15) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem 1.2rem !important;
}
div[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--gold-light) !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.4rem !important;
}

/* ── RESULT CARD ──────────────────────────────────────────── */
.result-wrap {
    background: linear-gradient(145deg, #1a1b22 0%, #0f100e 100%);
    border: 1px solid var(--gold);
    border-radius: var(--radius-lg);
    padding: 2.5rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 60px rgba(201,168,76,0.18), inset 0 1px 0 rgba(201,168,76,0.2);
    animation: fadeUp 0.5s ease forwards;
}
.result-wrap::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(201,168,76,0.1) 0%, transparent 65%);
    pointer-events: none;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.5rem;
}
.result-price {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 6vw, 3.8rem);
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0.3rem 0;
}
.result-price span { color: var(--gold); }
.result-sub {
    font-size: 0.88rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
}
.result-divider {
    width: 50px;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 1.2rem auto;
}
.result-chips {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.chip {
    background: var(--gold-dim);
    border: 1px solid var(--card-border);
    border-radius: 50px;
    padding: 5px 14px;
    font-size: 0.75rem;
    color: var(--gold-light);
    font-weight: 500;
}

/* ── SIDEBAR ──────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--card-bg) !important;
    border-right: 1px solid var(--card-border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-muted) !important; }
section[data-testid="stSidebar"] .stInfo {
    background: var(--surface) !important;
    border-left: 3px solid var(--gold) !important;
    border-radius: 0 var(--radius-md) var(--radius-md) 0 !important;
}

/* ── DIVIDER ──────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid rgba(201,168,76,0.15) !important;
    margin: 1.5rem 0 !important;
}

/* ── HELPER PILL STRIP ────────────────────────────────────── */
.stat-strip {
    display: flex;
    gap: 10px;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.stat-pill {
    background: var(--surface);
    border: 1px solid rgba(201,168,76,0.15);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.78rem;
    color: var(--text-muted);
    flex: 1;
    min-width: 120px;
    text-align: center;
}
.stat-pill strong {
    display: block;
    color: var(--gold-light);
    font-size: 1rem;
    font-family: 'Playfair Display', serif;
    margin-bottom: 2px;
}
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL PACKAGE ---
@st.cache_resource
def load_model():
    with open("bengaluru_house_model.pkl", "rb") as f:
        return pickle.load(f)

package = load_model()
model   = package["model"]
scaler  = package["scaler"]
features = package["features"]

# ── HERO BANNER ───────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">Real Estate Intelligence · Bengaluru</div>
  <div class="hero-title">Estimate Your Property's<br><span>True Market Value</span></div>
  <div class="hero-subtitle">
    Powered by an XGBoost model trained on thousands of Bengaluru listings.
    Fill in the property details below to receive an instant valuation.
  </div>
  <div class="hero-badge">XGBoost Model &nbsp;·&nbsp; R² ≈ 60%</div>
</div>
""", unsafe_allow_html=True)

# ── LIVE STAT STRIP (derived from inputs, updates after predict) ──
st.markdown("""
<div class="section-label">Property Specifications</div>
""", unsafe_allow_html=True)

# ── INPUT CARD — SIZE & LAYOUT ────────────────────────────────
st.markdown('<div class="input-card">', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    total_sqft = st.number_input("Total Area (sqft)", min_value=300, max_value=50000, value=1200, step=50)
with col2:
    area_sqft = st.number_input("Built-up Area (sqft)", min_value=300, max_value=50000, value=1100, step=50)
st.markdown('</div>', unsafe_allow_html=True)

# ── INPUT CARD — ROOMS ────────────────────────────────────────
st.markdown('<div class="input-card">', unsafe_allow_html=True)
col3, col4, col5 = st.columns(3)
with col3:
    bedroom_count = st.slider("BHK", 1, 10, 2)
with col4:
    bathrooms = st.slider("Bathrooms", 1, 10, 2)
with col5:
    balcony = st.selectbox("Balconies", [0, 1, 2, 3])
st.markdown('</div>', unsafe_allow_html=True)

# ── DERIVED STATS STRIP ───────────────────────────────────────
bath_to_room_ratio = bathrooms / bedroom_count if bedroom_count > 0 else 0
price_per_sqft_hint = "—"  # filled after prediction

eff = round(area_sqft / total_sqft * 100, 1) if total_sqft > 0 else 0
st.markdown(f"""
<div class="stat-strip">
  <div class="stat-pill"><strong>{total_sqft:,} sqft</strong>Total Area</div>
  <div class="stat-pill"><strong>{bedroom_count} BHK</strong>Configuration</div>
  <div class="stat-pill"><strong>{eff}%</strong>Built-up Ratio</div>
  <div class="stat-pill"><strong>{bath_to_room_ratio:.2f}</strong>Bath / BHK</div>
</div>
""", unsafe_allow_html=True)

# ── PREDICT BUTTON ────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("✦ Generate Valuation")

# ── RESULT ───────────────────────────────────────────────────
if predict_clicked:
    input_data = pd.DataFrame([[
        np.log(total_sqft),
        balcony,
        bedroom_count,
        area_sqft,
        bath_to_room_ratio
    ]], columns=features)

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    lakhs_val    = prediction * 100
    ppsf         = (lakhs_val * 100_000) / total_sqft if total_sqft > 0 else 0

    st.balloons()
    st.markdown(f"""
    <div class="result-wrap">
      <div class="result-label">Estimated Market Price</div>
      <div class="result-price"><span>₹</span> {prediction:.2f} <span style="font-size:0.45em;vertical-align:middle;color:#8A8678;">Crores</span></div>
      <div class="result-divider"></div>
      <div class="result-sub">≈ ₹ {lakhs_val:,.1f} Lakhs &nbsp;·&nbsp; ₹ {ppsf:,.0f} per sqft</div>
      <div class="result-chips">
        <span class="chip">🛏 {bedroom_count} BHK</span>
        <span class="chip">🚿 {bathrooms} Bath</span>
        <span class="chip">📐 {total_sqft:,} sqft</span>
        <span class="chip">🏙 Bengaluru</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Price / sqft", f"₹ {ppsf:,.0f}")
    m2.metric("In Lakhs",     f"₹ {lakhs_val:,.1f}L")
    m3.metric("Built-up",     f"{area_sqft:,} sqft")

# ── SIDEBAR ───────────────────────────────────────────────────
st.sidebar.markdown("### About This Model")
st.sidebar.info(
    "**Algorithm**: XGBoost Regressor\n\n"
    "**Accuracy**: R² ≈ 60%\n\n"
    "**Training data**: Historical Bengaluru listings\n\n"
    "Prices are indicative estimates. "
    "Consult a licensed property agent for official valuations."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Features used**")
for f in features:
    st.sidebar.markdown(f"• `{f}`")