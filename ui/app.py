#!/usr/bin/env python3
"""
Heart Disease Prediction App (Modern UI)
=======================================
A single-page, dark-themed Streamlit app for heart disease risk prediction with glassmorphism, animations, and advanced UX.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
from pathlib import Path

# --- Model input schema ---
FEATURES = [
    ('age', 'Age (years)', 'slider', {'min_value': 18, 'max_value': 100, 'value': 50, 'step': 1, 'help': 'Your age in years'}),
    ('sex', 'Biological Sex', 'select', {'options': [0, 1], 'format_func': lambda x: "👩 Female" if x == 0 else "👨 Male", 'help': 'Biological sex affects risk patterns'}),
    ('chest_pain_type', 'Chest Pain Type', 'select', {'options': [0, 1, 2, 3], 'format_func': lambda x: ["💔 Typical Angina", "💛 Atypical Angina", "💙 Non-Anginal Pain", "💚 Asymptomatic"][x], 'help': 'Type of chest pain'}),
    ('resting_bp_s', 'Resting Blood Pressure (mmHg)', 'slider', {'min_value': 90, 'max_value': 200, 'value': 130, 'step': 1, 'help': 'Systolic BP at rest'}),
    ('cholesterol', 'Cholesterol (mg/dl)', 'slider', {'min_value': 100, 'max_value': 600, 'value': 200, 'step': 1, 'help': 'Serum cholesterol level'}),
    ('fasting_blood_sugar', 'Fasting Blood Sugar > 120 mg/dl', 'select', {'options': [0, 1], 'format_func': lambda x: "❌ No" if x == 0 else "✅ Yes", 'help': 'Is fasting blood sugar above 120 mg/dl?'}),
    ('resting_ecg', 'Resting ECG Results', 'select', {'options': [0, 1, 2], 'format_func': lambda x: ["💚 Normal", "💛 ST-T Abnormality", "❤️ LV Hypertrophy"][x], 'help': 'ECG at rest'}),
    ('max_heart_rate', 'Maximum Heart Rate', 'slider', {'min_value': 60, 'max_value': 202, 'value': 150, 'step': 1, 'help': 'Max HR during exercise'}),
    ('exercise_angina', 'Exercise Induced Angina', 'select', {'options': [0, 1], 'format_func': lambda x: "❌ No" if x == 0 else "✅ Yes", 'help': 'Chest pain during exercise?'}),
    ('oldpeak', 'ST Depression', 'slider', {'min_value': 0.0, 'max_value': 6.2, 'value': 0.0, 'step': 0.1, 'help': 'ST depression by exercise'}),
    ('st_slope', 'Slope of Peak Exercise ST', 'select', {'options': [0, 1, 2], 'format_func': lambda x: ["📈 Upsloping", "➡️ Flat", "📉 Downsloping"][x], 'help': 'Slope of peak exercise ST segment'})
]

# --- Normal ranges for tooltips ---
NORMAL_RANGES = {
    'age': 'Normal: 18-100 years',
    'sex': '0 = Female, 1 = Male',
    'chest_pain_type': '0 = Typical Angina, 1 = Atypical Angina, 2 = Non-Anginal Pain, 3 = Asymptomatic',
    'resting_bp_s': 'Normal: 90-120 mmHg',
    'cholesterol': 'Normal: 125-200 mg/dl',
    'fasting_blood_sugar': '0 = No, 1 = Yes (Normal: ≤120 mg/dl)',
    'resting_ecg': '0 = Normal, 1 = ST-T Abnormality, 2 = LV Hypertrophy',
    'max_heart_rate': 'Normal: 100-190 bpm',
    'exercise_angina': '0 = No, 1 = Yes',
    'oldpeak': 'Normal: 0.0-2.0',
    'st_slope': '0 = Upsloping, 1 = Flat, 2 = Downsloping'
}

# --- Custom CSS for dark theme, glassmorphism, and animations ---
try:
    with open(os.path.join(os.path.dirname(__file__), 'assets', 'style.css')) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass  # or st.warning("Custom CSS file not found.")

# --- Page config ---
st.set_page_config(
    page_title="❤️ Heart Disease AI Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Responsive device detection (mobile/tablet/desktop) ---
# Instead of device detection, just use columns for desktop/tablet
# (Removed duplicate input block here to avoid errors)

# --- Model loading (cache for performance) ---
@st.cache_resource

def load_model():
    model_path = Path(__file__).parent.parent / 'models' / 'heart_disease_pipeline.pkl'
    try:
        with st.spinner('🚀 Loading AI model for heart disease prediction...'):
            if not model_path.exists():
                st.error("❌ Model file not found! Please ensure 'models/heart_disease_pipeline.pkl' exists.")
                st.stop()
            model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load the prediction model: {e}\nPlease contact support or check the model file.")
        st.stop()

model = load_model()

# --- Hero Section (Enhanced Responsive + Gradient BG) ---
st.markdown('''
<style>
/* Animated gradient background for the whole app */
body, .stApp {
    background: linear-gradient(120deg, #232526 0%, #7f5af0 50%, #00e6fe 100%);
    background-size: 200% 200%;
    animation: gradientMove 8s ease-in-out infinite;
}
@keyframes gradientMove {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}

.hero-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 420px;
    margin-top: 1.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    z-index: 2;
}
.hero-glass-card {
    background: linear-gradient(135deg, rgba(127,90,240,0.35) 0%, rgba(0,230,254,0.22) 100%), rgba(36, 37, 46, 0.45);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 2.5rem;
    border: 1.5px solid rgba(127, 90, 240, 0.18);
    padding: 2.5rem 3.5rem 2rem 3.5rem;
    text-align: center;
    max-width: 650px;
    margin: 0 auto;
    position: relative;
    z-index: 2;
    transition: box-shadow 0.3s;
}
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    color: #fff;
    letter-spacing: 1.5px;
    margin-bottom: 0.5rem;
    text-shadow: 0 2px 16px #7f5af0;
}
.hero-subtitle {
    font-size: 1.35rem;
    color: #cfcfff;
    font-weight: 400;
    margin-bottom: 2.2rem;
}
.hero-stats {
    display: flex;
    justify-content: center;
    gap: 2.2rem;
    margin-bottom: 1.5rem;
}
.stat-item {
    background: rgba(127, 90, 240, 0.13);
    border-radius: 1.2rem;
    padding: 1.1rem 2.1rem;
    box-shadow: 0 2px 16px #7f5af0;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 100px;
    border: 1.5px solid rgba(127, 90, 240, 0.18);
    transition: transform 0.2s;
}
.stat-item:hover {
    transform: scale(1.07) rotate(-2deg);
    box-shadow: 0 4px 32px #7f5af0;
}
.stat-number {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffe066;
    margin-bottom: 0.2rem;
    text-shadow: 0 2px 8px #7f5af0;
}
.stat-label {
    font-size: 1.05rem;
    color: #fff;
    font-weight: 500;
    letter-spacing: 0.5px;
}
.hero-wave {
    position: absolute;
    left: 0; right: 0; bottom: -1px;
    width: 100%;
    height: 70px;
    z-index: 1;
    overflow: hidden;
}
.hero-wave svg {
    width: 100%;
    height: 100%;
    display: block;
}

/* Responsive styles for mobile */
@media (max-width: 700px) {
  .hero-glass-card {
    padding: 1.2rem 0.7rem 1.2rem 0.7rem;
    max-width: 98vw;
    border-radius: 1.2rem;
  }
  .hero-title {
    font-size: 2rem;
  }
  .hero-subtitle {
    font-size: 1rem;
  }
  .hero-stats {
    flex-direction: column;
    gap: 1.1rem;
  }
  .stat-item {
    min-width: 80px;
    padding: 0.7rem 1.1rem;
  }
}
</style>
<div class="hero-section">
  <div class="hero-glass-card">
    <h1 class="hero-title">❤️🤖AI Heart Disease Predictor</h1>
    <p class="hero-subtitle">Advanced Machine Learning for Cardiovascular Risk Assessment<br/>
      <span style="color:#7f5af0; font-weight:600;">Instant. Private. Personalized.</span>
    </p>
    <div class="hero-stats">
      <div class="stat-item">
        <span class="stat-number">92%</span>
        <div class="stat-label">Accuracy</div>
      </div>
      <div class="stat-item">
        <span class="stat-number">97%</span>
        <div class="stat-label">AUC Score</div>
      </div>
      <div class="stat-item">
        <span class="stat-number">11</span>
        <div class="stat-label">Features</div>
      </div>
    </div>
  </div>
  <div class="hero-wave">
    <svg viewBox="0 0 1440 70" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M0,30 C360,90 1080,-30 1440,30 L1440,70 L0,70 Z" fill="#7f5af0" fill-opacity="0.18"></path>
      <path d="M0,50 C400,0 1040,100 1440,50 L1440,70 L0,70 Z" fill="#00e6fe" fill-opacity="0.10"></path>
    </svg>
  </div>
</div>
''', unsafe_allow_html=True)

# --- Input Form ---
# Enhanced: Responsive, professional, with left padding and improved titles
st.markdown('''
<div class="form-glass-card">
  <div class="form-section-title">
    <span style="font-size:2.2rem;">🔬🤖</span> AI Clinical Assessment
  </div>
  <div class="wave" style="height:40px;"></div>
</div>
''', unsafe_allow_html=True)

with st.form("prediction_form", clear_on_submit=False):
    # Responsive: 1 column on mobile/tablet, 2 on desktop
    cols = st.columns(1) if st.session_state.get('is_mobile', False) or st.session_state.get('is_tablet', False) else st.columns(2)
    user_input = {}
    for idx, (key, label, kind, params) in enumerate(FEATURES):
        # Enhanced label: bold, icon, left padding, AI emoji for tech features
        icon = ''
        if 'age' in key: icon = '🎂'
        elif 'sex' in key: icon = '⚧️'
        elif 'chest_pain' in key: icon = '💔'
        elif 'bp' in key: icon = '🩸'
        elif 'cholesterol' in key: icon = '🧬'
        elif 'sugar' in key: icon = '🍬'
        elif 'ecg' in key: icon = '🩺'
        elif 'heart_rate' in key: icon = '❤️'
        elif 'angina' in key: icon = '🏃'
        elif 'oldpeak' in key: icon = '📉'
        elif 'slope' in key: icon = '📈'
        # Add AI emoji for all labels
        pretty_label = f'<span class="input-label">🤖 {icon} {label}</span>'
        # Compose tooltip with normal range
        help_text = params.get('help', '')
        normal_range = NORMAL_RANGES.get(key, '')
        tooltip = f"{help_text}"
        if normal_range:
            tooltip += f"\nNormal range: {normal_range}"
        with cols[idx % len(cols)]:
            st.markdown(pretty_label, unsafe_allow_html=True)
            if kind == 'slider':
                user_input[key] = st.slider(
                    label, 
                    **{k: v for k, v in params.items() if k != 'help'}, 
                    help=tooltip,
                    format=None,
                    key=key
                )
            elif kind == 'select':
                selectbox_kwargs = {k: v for k, v in params.items() if k != 'help' and k != 'format_func'}
                if 'format_func' in params:
                    user_input[key] = st.selectbox(
                        label,
                        **selectbox_kwargs,
                        help=tooltip,
                        format_func=params['format_func'],
                        key=key
                    )
                else:
                    user_input[key] = st.selectbox(
                        label,
                        **selectbox_kwargs,
                        help=tooltip,
                        key=key
                    )
    submitted = st.form_submit_button("🔮 Predict Heart Disease Risk", use_container_width=True)

# --- Prediction & Results ---
def predict(model, user_input):
    try:
        X = pd.DataFrame(
            [[user_input[k[0]] for k in FEATURES]],
            columns=[k[0] for k in FEATURES]  # type: ignore
        )
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]
        return pred, proba
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None, None

def risk_level(prob):
    if prob < 0.3:
        return "Low Risk", "💚", "low-risk"
    elif prob < 0.7:
        return "Medium Risk", "💛", "medium-risk"
    else:
        return "High Risk", "❤️", "high-risk"

def gauge_chart(prob):
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={'text': "Risk (%)", 'font': {'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': "#fd5c63" if prob > 0.7 else ("#ffe066" if prob > 0.3 else "#27ae60")},
            'steps': [
                {'range': [0, 30], 'color': "rgba(39, 174, 96, 0.5)"},
                {'range': [30, 70], 'color': "rgba(243, 156, 18, 0.5)"},
                {'range': [70, 100], 'color': "rgba(231, 76, 60, 0.5)"}
            ]
        }
    )).update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))

def radar_chart(user_input):
    radar_features = ['age', 'resting_bp_s', 'cholesterol', 'max_heart_rate', 'oldpeak']
    values = [user_input[k] for k in radar_features]
    max_vals = [100, 200, 600, 202, 6.2]
    norm = [v/m for v, m in zip(values, max_vals)]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=norm, theta=["Age", "BP", "Chol", "Max HR", "ST Depress"], fill='toself', name='Profile', line_color='#667eea'))
    fig.update_layout(polar=dict(bgcolor='rgba(0,0,0,0.2)', radialaxis=dict(visible=True, range=[0,1], tickcolor='white')), showlegend=False, template="plotly_dark", height=350)
    return fig

def comparison_chart(user_input):
    data_path = Path(__file__).parent.parent / 'data' / 'heart_disease_selected.csv'
    if not data_path.exists():
        return None
    df = pd.read_csv(data_path)
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=['18-30', '31-45', '46-60', '60+'])
    risk_by_age = df.groupby('age_group', observed=False)['target'].mean()
    user_age = user_input['age']
    if user_age <= 30:
        user_group = '18-30'
    elif user_age <= 45:
        user_group = '31-45'
    elif user_age <= 60:
        user_group = '46-60'
    else:
        user_group = '60+'
    fig = go.Figure()
    fig.add_trace(go.Bar(x=risk_by_age.index, y=risk_by_age.values, name='Avg Risk', marker_color='#00e6fe'))
    fig.add_trace(go.Scatter(x=[user_group], y=[risk_by_age.loc[user_group]], mode='markers', name='You', marker=dict(size=15, color='red')))
    fig.update_layout(template="plotly_dark", height=350, title="Risk by Age Group", xaxis_title="Age Group", yaxis_title="Risk Probability")
    return fig

# --- Results Section ---
if submitted:
    with st.spinner('🤖✨ Our AI is analyzing your data...'):
        import time
        # Show custom loader only while waiting
        loading_placeholder = st.empty()
        loading_placeholder.markdown('''<div style="text-align:center; margin-top:2rem;">
            <div class="loader" style="position:relative; display:inline-block;">
                <span style="position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); font-size:2.2rem;">🤖</span>
            </div>
            <div style="font-size:1.2rem; color:#7f5af0; margin-top:1rem; font-weight:600; letter-spacing:0.5px;">AI is working on your personalized prediction...</div>
        </div>''', unsafe_allow_html=True)
        time.sleep(1.2)  # Short delay for effect
        pred, proba = predict(model, user_input)
        loading_placeholder.empty()  # Remove loader as soon as result is ready
        if pred is not None:
            risk, icon, risk_class = risk_level(proba)
            st.markdown(f'''<div class="prediction-result {risk_class}"><div class="risk-icon">{icon}</div><div class="risk-title">{risk}</div><div class="risk-probability">Probability: {proba:.1%}</div></div>''', unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(proba), use_container_width=True)
            st.markdown("### 🎯 Confidence Level")
            if proba is not None:
                st.progress(proba)
            st.caption(f"Model confidence: {proba:.1%}")
            st.markdown('<div class="glass-card"><h3>📊 Health Profile Analysis</h3></div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(radar_chart(user_input), use_container_width=True)
            with col2:
                comp_fig = comparison_chart(user_input)
                if comp_fig:
                    st.plotly_chart(comp_fig, use_container_width=True)
            st.markdown('<div class="glass-card"><h3>💡 Personalized Recommendations</h3></div>', unsafe_allow_html=True)
            if pred == 1:
                st.markdown('''<div class="recommendations"><ul><li><b>Immediate Action:</b> Consult a cardiologist</li><li><b>Lifestyle:</b> Adopt a heart-healthy diet</li><li><b>Exercise:</b> Start with light activity</li><li><b>Monitoring:</b> Regular BP checks</li><li><b>Medication:</b> Follow prescribed treatments</li></ul></div>''', unsafe_allow_html=True)
            else:
                st.markdown('''<div class="recommendations"><ul><li><b>Maintenance:</b> Continue healthy lifestyle</li><li><b>Prevention:</b> Regular checkups</li><li><b>Exercise:</b> Maintain activity</li><li><b>Diet:</b> Balanced nutrition</li><li><b>Monitoring:</b> Annual assessments</li></ul></div>''', unsafe_allow_html=True)
            st.markdown('''<div class="disclaimer"><h4>⚠️ Medical Disclaimer</h4><p>This tool is for educational purposes only. Always consult a healthcare professional for medical advice.</p></div>''', unsafe_allow_html=True)

# --- End of file --- 