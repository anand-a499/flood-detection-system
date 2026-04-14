
import streamlit as st
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(page_title="FloodWatch AI", page_icon="🌊", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "flood_risk_model.joblib"
model = joblib.load(MODEL_PATH)

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}

ACTIONS = {
    "Low": [
        "Monitor IMD or local weather updates every few hours.",
        "Keep emergency torch, power bank, and contacts ready.",
        "Avoid unnecessary travel during intense rain."
    ],
    "Medium": [
        "Prepare an emergency kit with medicines, water, and documents.",
        "Move valuables and electronics to higher shelves.",
        "Avoid waterlogged roads, underpasses, and riverbank areas."
    ],
    "High": [
        "Shift family members, vehicles, and valuables to safer higher locations.",
        "Stay in touch with local authorities and neighbors.",
        "Be ready to evacuate immediately if rainfall or water level rises further."
    ],
    "Critical": [
        "Evacuate low-lying areas immediately.",
        "Switch off electricity if water enters the building.",
        "Move to official shelter / safe elevated location and follow authority instructions."
    ]
}

ANALYSIS = {
    "Low": "Current conditions suggest limited immediate flood threat, but continued monitoring is important because local drainage and rainfall can change quickly.",
    "Medium": "The model detects growing flood pressure from combined environmental factors. Preparedness actions should begin now to reduce risk escalation.",
    "High": "The model identifies strong flood-driving conditions. Immediate protective actions are recommended, especially in low-lying or poorly drained areas.",
    "Critical": "The system detects severe flood conditions with high operational risk. Immediate evacuation and emergency response measures are strongly advised."
}

st.markdown("""
<style>
.stApp {background: linear-gradient(180deg,#07101f 0%, #0b1730 100%); color: #e8f4ff;}
.block-container {padding-top: 1.2rem; max-width: 1250px;}
.card {
    background: rgba(10,20,40,0.92);
    border: 1px solid #17345e;
    border-radius: 20px;
    padding: 22px;
    box-shadow: 0 0 0 1px rgba(0,212,255,0.03), 0 18px 40px rgba(0,0,0,0.25);
}
.big-title {font-size: 2.4rem; font-weight: 800; color: #e8f4ff; margin-bottom: 0.2rem;}
.subtle {color: #7ea0c7; font-size: 0.95rem;}
.metric-box {
    background: #121f3a; border: 1px solid #1e355c; border-radius: 14px; padding: 14px;
}
.section-label {
    color: #6ca4da; font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.6rem;
}
.risk-low {background: rgba(0,230,118,0.12); border: 1px solid rgba(0,230,118,0.45); color:#00e676; padding:14px 18px; border-radius:16px; font-size:1.6rem; font-weight:800;}
.risk-medium {background: rgba(255,179,0,0.12); border: 1px solid rgba(255,179,0,0.45); color:#ffb300; padding:14px 18px; border-radius:16px; font-size:1.6rem; font-weight:800;}
.risk-high {background: rgba(255,61,0,0.12); border: 1px solid rgba(255,61,0,0.45); color:#ff6d3d; padding:14px 18px; border-radius:16px; font-size:1.6rem; font-weight:800;}
.risk-critical {background: rgba(255,0,68,0.14); border: 1px solid rgba(255,0,68,0.55); color:#ff4d79; padding:14px 18px; border-radius:16px; font-size:1.6rem; font-weight:800;}
.small-note {color: #7ea0c7; font-size: 0.78rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🌊 FloodWatch AI Disaster Preparedness System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Multi-input flood risk assessment with trained model, severity score, confidence, measures, and response guidance.</div>', unsafe_allow_html=True)

preset = st.selectbox("Quick scenario preset", ["Custom", "Normal Day", "Heavy Rain", "Urban Flood Threat", "River Overflow", "Extreme Emergency"])

preset_values = {
    "Custom": [80, 6.0, 50, 50, 50, 70],
    "Normal Day": [20, 2.5, 35, 75, 30, 18],
    "Heavy Rain": [110, 5.0, 65, 45, 45, 130],
    "Urban Flood Threat": [95, 6.8, 70, 20, 60, 150],
    "River Overflow": [130, 9.5, 75, 35, 65, 180],
    "Extreme Emergency": [220, 11.0, 88, 10, 80, 260],
}
default_vals = preset_values[preset]

left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Flood Risk Parameters</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        rainfall = st.slider("Rainfall (mm/h)", 0.0, 300.0, float(default_vals[0]), 1.0)
        soil = st.slider("Soil Moisture (%)", 0.0, 100.0, float(default_vals[2]), 1.0)
        elevation = st.slider("Elevation Risk (0-100)", 0.0, 100.0, float(default_vals[4]), 1.0)
    with c2:
        river = st.slider("River Level (m)", 0.0, 12.0, float(default_vals[1]), 0.1)
        drainage = st.slider("Drainage Index (0-100)", 0.0, 100.0, float(default_vals[3]), 1.0)
        prev24 = st.slider("24h Rainfall (mm)", 0.0, 300.0, float(default_vals[5]), 1.0)

    st.markdown('<div class="small-note">Multiple input options are supported through sliders plus quick scenario presets above.</div>', unsafe_allow_html=True)
    run = st.button("⚡ Run Flood Risk Analysis", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Prediction Output</div>', unsafe_allow_html=True)

    features = np.array([[rainfall, river, soil, drainage, elevation, prev24]], dtype=float)
    probs = model.predict_proba(features)[0]
    pred = int(model.predict(features)[0])
    risk = RISK_LABELS[pred]
    confidence = float(np.max(probs)) * 100
    severity = round((probs[1]*35 + probs[2]*70 + probs[3]*100 + probs[0]*15), 1)

    cls = {"Low":"risk-low","Medium":"risk-medium","High":"risk-high","Critical":"risk-critical"}[risk]
    st.markdown(f'<div class="{cls}">{risk} Risk Level</div>', unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown('<div class="metric-box"><div class="section-label">Severity Score</div><div style="font-size:1.6rem;font-weight:800;">{:.1f}/100</div></div>'.format(severity), unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-box"><div class="section-label">Model Confidence</div><div style="font-size:1.6rem;font-weight:800;">{:.1f}%</div></div>'.format(confidence), unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Low", f"{probs[0]*100:.1f}%")
    p2.metric("Medium", f"{probs[1]*100:.1f}%")
    p3.metric("High", f"{probs[2]*100:.1f}%")
    p4.metric("Critical", f"{probs[3]*100:.1f}%")

    st.progress(min(int(severity), 100), text=f"Risk score: {severity}/100")

    st.markdown("### Immediate Response Measures")
    for item in ACTIONS[risk]:
        st.write(f"- {item}")

    st.markdown("### AI-style Analysis")
    st.info(ANALYSIS[risk])

    st.markdown("### Input Summary")
    st.write({
        "rainfall_mm_h": rainfall,
        "river_level_m": river,
        "soil_moisture_pct": soil,
        "drainage_index": drainage,
        "elevation_risk": elevation,
        "prev_24h_rainfall_mm": prev24
    })
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("**Prototype note:** This package includes a pre-trained `flood_risk_model.joblib` so it works immediately after download.")
