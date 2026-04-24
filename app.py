import streamlit as st
import numpy as np
import pandas as pd
import pickle

model      = pickle.load(open('model.pkl',         'rb'))
scaler     = pickle.load(open('scaler.pkl',        'rb'))
fe_scaler  = pickle.load(open('fe_scaler.pkl',     'rb'))
feat_names = pickle.load(open('feature_names.pkl', 'rb'))

st.set_page_config(page_title="BookingGuard AI", page_icon="🏨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: #f8faff; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1400px !important; }

/* Hide Streamlit default UI */
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] { display: none; }
.stDeployButton { display: none; }
div[data-testid="stStatusWidget"] { display: none; }
#root > div:first-child { padding-top: 0 !important; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    border-radius: 28px;
    padding: 52px 48px;
    margin-bottom: 36px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(99,102,241,0.35);
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 300px; height: 300px;
    background: rgba(255,255,255,0.08);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 200px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.2);
    backdrop-filter: blur(10px);
    color: white;
    padding: 6px 16px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.3);
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: white;
    line-height: 1.1;
    margin-bottom: 14px;
    text-shadow: 0 2px 20px rgba(0,0,0,0.2);
}
.hero-sub { color: rgba(255,255,255,0.8); font-size: 1.05rem; line-height: 1.7; font-weight: 400; }
.stats-row { display: flex; gap: 0; margin-top: 36px; }
.stat-box {
    flex: 1;
    text-align: center;
    padding: 20px;
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    margin-right: 12px;
    border: 1px solid rgba(255,255,255,0.2);
}
.stat-box:last-child { margin-right: 0; }
.stat-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: white;
    display: block;
}
.stat-lbl { font-size: 11px; color: rgba(255,255,255,0.7); text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; display: block; }

/* Section Labels */
.sec-label {
    font-size: 12px;
    font-weight: 700;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-label::before {
    content: '';
    width: 24px; height: 3px;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    border-radius: 2px;
    display: inline-block;
}

/* Cards */
.input-card {
    background: white;
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 16px;
    border: 1.5px solid #e8eaf6;
    box-shadow: 0 4px 20px rgba(99,102,241,0.06);
    transition: box-shadow 0.2s, border-color 0.2s;
}
.input-card:hover { box-shadow: 0 8px 30px rgba(99,102,241,0.12); border-color: #c5cae9; }
.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1.5px solid #f0f2ff;
}
.card-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}
.card-title-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px;
    font-weight: 600;
    color: #1e1b4b;
}

/* Predict Button */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 18px 32px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 35px rgba(99,102,241,0.5) !important;
}

/* Results */
.result-wrap {
    border-radius: 24px;
    padding: 44px 40px;
    text-align: center;
    margin: 20px 0;
    position: relative;
    overflow: hidden;
}
.result-cancel {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border: 2px solid #fecdd3;
    box-shadow: 0 16px 50px rgba(239,68,68,0.12);
}
.result-safe {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 2px solid #bbf7d0;
    box-shadow: 0 16px 50px rgba(34,197,94,0.12);
}
.result-emoji { font-size: 3.5rem; margin-bottom: 12px; display: block; }
.result-headline {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 8px;
}
.result-cancel .result-headline { color: #dc2626; }
.result-safe .result-headline   { color: #16a34a; }
.result-desc { font-size: 1rem; color: #6b7280; font-weight: 400; }

/* Risk Pill */
.risk-pill {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 100px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin: 20px 0 0 0;
}
.risk-high   { background: #fef2f2; color: #dc2626; border: 2px solid #fca5a5; }
.risk-medium { background: #fffbeb; color: #d97706; border: 2px solid #fcd34d; }
.risk-low    { background: #f0fdf4; color: #16a34a; border: 2px solid #86efac; }

/* Metrics Row */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 24px 0;
}
.metric-card {
    background: white;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    border: 1.5px solid #e8eaf6;
    box-shadow: 0 4px 15px rgba(99,102,241,0.06);
}
.metric-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: block;
}
.metric-lbl { font-size: 12px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

/* Prob Bars */
.prob-section {
    background: white;
    border-radius: 20px;
    padding: 24px 28px;
    border: 1.5px solid #e8eaf6;
    margin: 16px 0;
    box-shadow: 0 4px 15px rgba(99,102,241,0.06);
}
.prob-row { margin-bottom: 20px; }
.prob-row:last-child { margin-bottom: 0; }
.prob-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.prob-label { font-size: 13px; font-weight: 600; color: #374151; }
.prob-pct { font-family: 'Space Grotesk', sans-serif; font-size: 15px; font-weight: 700; }
.prob-cancel-pct { color: #dc2626; }
.prob-safe-pct   { color: #16a34a; }
.bar-bg { background: #f1f5f9; border-radius: 100px; height: 10px; overflow: hidden; }
.bar-cancel { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #f87171, #dc2626); transition: width 1s ease; }
.bar-safe   { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #4ade80, #16a34a); transition: width 1s ease; }

/* Factor Grid */
.factor-section {
    background: white;
    border-radius: 20px;
    padding: 24px 28px;
    border: 1.5px solid #e8eaf6;
    box-shadow: 0 4px 15px rgba(99,102,241,0.06);
}
.factor-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 16px; }
.factor-box {
    background: linear-gradient(135deg, #f8f9ff, #f0f2ff);
    border-radius: 14px;
    padding: 18px 14px;
    text-align: center;
    border: 1.5px solid #e0e7ff;
}
.factor-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #4f46e5;
    display: block;
    margin-bottom: 4px;
}
.factor-lbl { font-size: 11px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.8px; }

/* Streamlit widget overrides */
.stSelectbox label, .stNumberInput label, .stSlider label {
    color: #374151 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stMetricValue"] {
    color: #4f46e5 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #9ca3af !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
.stDivider { border-color: #e8eaf6 !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ AI Powered Prediction System</div>
    <div class="hero-title">BookingGuard AI</div>
    <div class="hero-sub">
        Intelligent hotel booking cancellation prediction using Logistic Regression.<br>
        Analyse booking patterns and predict cancellation risk instantly.
    </div>
    <div class="stats-row">
        <div class="stat-box">
            <span class="stat-num">71.21%</span>
            <span class="stat-lbl">Accuracy</span>
        </div>
        <div class="stat-box">
            <span class="stat-num">76.27%</span>
            <span class="stat-lbl">Recall</span>
        </div>
        <div class="stat-box">
            <span class="stat-num">80.31%</span>
            <span class="stat-lbl">AUC-ROC</span>
        </div>
        <div class="stat-box">
            <span class="stat-num">87K+</span>
            <span class="stat-lbl">Records</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────
st.markdown('<div class="sec-label">Booking Information</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="input-card"><div class="card-header"><div class="card-icon">👤</div><div class="card-title-text">Guest Profile</div></div></div>', unsafe_allow_html=True)
    hotel         = st.selectbox("Hotel Type", options=[0,1], format_func=lambda x: "City Hotel" if x==0 else "Resort Hotel")
    adults        = st.number_input("Adults",         min_value=1, max_value=10, value=2)
    children      = st.number_input("Children",       min_value=0, max_value=10, value=0)
    is_repeated   = st.selectbox("Repeated Guest",    options=[0,1], format_func=lambda x: "No" if x==0 else "Yes")
    customer_type = st.selectbox("Customer Type",     options=[0,1,2,3], format_func=lambda x: ['Contract','Group','Transient','Transient-Party'][x])

with col2:
    st.markdown('<div class="input-card"><div class="card-header"><div class="card-icon">📅</div><div class="card-title-text">Stay Details</div></div></div>', unsafe_allow_html=True)
    lead_time       = st.slider("Lead Time (days)",   0, 500, 90)
    weekend_nights  = st.number_input("Weekend Nights",  min_value=0, max_value=20, value=2)
    week_nights     = st.number_input("Week Nights",     min_value=0, max_value=30, value=5)
    booking_changes = st.number_input("Booking Changes", min_value=0, max_value=20, value=0)
    special_req     = st.number_input("Special Requests",min_value=0, max_value=10, value=0)

with col3:
    st.markdown('<div class="input-card"><div class="card-header"><div class="card-icon">💳</div><div class="card-title-text">Payment & Market</div></div></div>', unsafe_allow_html=True)
    deposit_type   = st.selectbox("Deposit Type",      options=[0,1,2], format_func=lambda x: ['No Deposit','Non Refundable','Refundable'][x])
    adr            = st.number_input("Avg Daily Rate ($)", min_value=0.0, max_value=1000.0, value=100.0)
    prev_cancel    = st.number_input("Previous Cancellations", min_value=0, max_value=20, value=0)
    parking        = st.number_input("Car Parking Spaces",     min_value=0, max_value=5,  value=0)
    market_segment = st.selectbox("Market Segment",   options=[0,1,2,3,4,5,6], format_func=lambda x: ['Aviation','Complementary','Corporate','Direct','Groups','Offline TA','Online TA'][x])

st.markdown("<br>", unsafe_allow_html=True)
predict = st.button("🔮  PREDICT CANCELLATION RISK", type="primary", use_container_width=True)

if predict:
    input_dict = {col: 0 for col in feat_names}
    input_dict.update({
        'hotel': hotel, 'lead_time': lead_time,
        'arrival_date_year': 2017, 'arrival_date_month': 8,
        'arrival_date_week_number': 32, 'arrival_date_day_of_month': 15,
        'stays_in_weekend_nights': weekend_nights, 'stays_in_week_nights': week_nights,
        'adults': adults, 'children': children, 'is_repeated_guest': is_repeated,
        'previous_cancellations': prev_cancel, 'booking_changes': booking_changes,
        'deposit_type': deposit_type, 'adr': adr,
        'total_of_special_requests': special_req,
        'required_car_parking_spaces': parking,
        'market_segment': market_segment, 'customer_type': customer_type
    })

    input_df     = pd.DataFrame([input_dict])[feat_names]
    input_scaled = scaler.transform(input_df)
    idx = {col: i for i, col in enumerate(feat_names)}

    f1 = input_scaled[:,idx['stays_in_weekend_nights']] + input_scaled[:,idx['stays_in_week_nights']]
    f2 = input_scaled[:,idx['lead_time']] * input_scaled[:,idx['previous_cancellations']]
    f3 = input_scaled[:,idx['total_of_special_requests']] + input_scaled[:,idx['booking_changes']]
    f4 = input_scaled[:,idx['lead_time']] / (input_scaled[:,idx['total_of_special_requests']] + 1)

    new_feats        = np.column_stack([f1,f2,f3,f4])
    new_feats_scaled = fe_scaler.transform(new_feats)
    input_fe         = np.column_stack([input_scaled, new_feats_scaled])

    prediction  = model.predict(input_fe)[0]
    probability = model.predict_proba(input_fe)[0]
    cancel_pct  = probability[1] * 100
    stay_pct    = probability[0] * 100

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Prediction Result</div>', unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(f"""
        <div class="result-wrap result-cancel">
            <span class="result-emoji">⚠️</span>
            <div class="result-headline">Booking Will Be Cancelled</div>
            <div class="result-desc">This booking has a high cancellation risk based on the provided details</div>
            <div><span class="risk-pill {'risk-high' if cancel_pct>80 else 'risk-medium'}">
                {'🔴 HIGH RISK' if cancel_pct>80 else '🟡 MEDIUM RISK' if cancel_pct>60 else '🟡 LOW-MEDIUM RISK'}
            </span></div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-wrap result-safe">
            <span class="result-emoji">✅</span>
            <div class="result-headline">Booking Will Not Be Cancelled</div>
            <div class="result-desc">This booking is likely to stay confirmed based on the provided details</div>
            <div><span class="risk-pill risk-low">🟢 {'LOW RISK' if stay_pct>80 else 'MODERATE CONFIDENCE'}</span></div>
        </div>""", unsafe_allow_html=True)

    # Metrics
    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-card">
            <span class="metric-num">{cancel_pct:.1f}%</span>
            <div class="metric-lbl">Cancel Probability</div>
        </div>
        <div class="metric-card">
            <span class="metric-num">{stay_pct:.1f}%</span>
            <div class="metric-lbl">Stay Probability</div>
        </div>
        <div class="metric-card">
            <span class="metric-num">{max(probability)*100:.1f}%</span>
            <div class="metric-lbl">Confidence</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Prob Bars
    st.markdown(f"""
    <div class="prob-section">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:600;
                    color:#374151;margin-bottom:20px;">Probability Breakdown</div>
        <div class="prob-row">
            <div class="prob-header">
                <span class="prob-label">Cancellation Risk</span>
                <span class="prob-pct prob-cancel-pct">{cancel_pct:.1f}%</span>
            </div>
            <div class="bar-bg"><div class="bar-cancel" style="width:{cancel_pct}%"></div></div>
        </div>
        <div class="prob-row">
            <div class="prob-header">
                <span class="prob-label">Stay Confidence</span>
                <span class="prob-pct prob-safe-pct">{stay_pct:.1f}%</span>
            </div>
            <div class="bar-bg"><div class="bar-safe" style="width:{stay_pct}%"></div></div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Factors
    dep_label = 'Non-Refund' if deposit_type==1 else 'Refundable' if deposit_type==2 else 'No Deposit'
    st.markdown(f"""
    <div class="factor-section">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:600;color:#374151;">
            Key Booking Factors
        </div>
        <div class="factor-grid">
            <div class="factor-box">
                <span class="factor-val">{lead_time}d</span>
                <span class="factor-lbl">Lead Time</span>
            </div>
            <div class="factor-box">
                <span class="factor-val">{dep_label}</span>
                <span class="factor-lbl">Deposit Type</span>
            </div>
            <div class="factor-box">
                <span class="factor-val">{prev_cancel}</span>
                <span class="factor-lbl">Past Cancels</span>
            </div>
            <div class="factor-box">
                <span class="factor-val">{special_req}</span>
                <span class="factor-lbl">Special Requests</span>
            </div>
            <div class="factor-box">
                <span class="factor-val">{'Yes' if is_repeated==1 else 'No'}</span>
                <span class="factor-lbl">Returning Guest</span>
            </div>
            <div class="factor-box">
                <span class="factor-val">${adr:.0f}</span>
                <span class="factor-lbl">Daily Rate</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.markdown("""
<div style='text-align:center;padding:12px 0;'>
    <span style='font-size:12px;color:#d1d5db;letter-spacing:1px;'>
        BookingGuard AI &nbsp;·&nbsp; Logistic Regression &nbsp;·&nbsp;
        Hotel Booking Cancellation Prediction &nbsp;·&nbsp; IT4060 Machine Learning
    </span>
</div>""", unsafe_allow_html=True)