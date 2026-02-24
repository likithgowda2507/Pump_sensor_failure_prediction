import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import datetime
import time

# --- CONFIGURATION ---
st.set_page_config(
    layout="wide",
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    initial_sidebar_state="expanded"
)

# --- MODERN INDUSTRIAL UI STYLING ---
def apply_custom_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        
        /* Global Background and Typography */
        .stApp {
            background-color: #0B0E14;
            color: #E0E6ED;
            font-family: 'Inter', sans-serif;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #12171F;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Glassmorphism Title Container */
        .glass-header {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header-title {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, #00C6FF 0%, #0072FF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }
        
        .header-subtitle {
            font-size: 1rem;
            color: #8B949E;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-top: 0.5rem;
        }
        
        /* Gradient KPI Cards */
        .kpi-container {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .kpi-card {
            background: linear-gradient(135deg, #1E2530 0%, #12171F 100%);
            border-radius: 20px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .kpi-card:hover {
            transform: translateY(-5px);
            border: 1px solid rgba(0, 198, 255, 0.4);
            box-shadow: 0 15px 40px rgba(0, 198, 255, 0.1);
        }
        
        .kpi-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #00C6FF, #0072FF);
            opacity: 0.6;
        }
        
        .kpi-label {
            font-size: 0.8rem;
            color: #8B949E;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
        }
        
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #FFFFFF;
            line-height: 1.2;
            margin-top: 0.2rem;
        }
        
        .kpi-small {
            font-size: 1.4rem;
        }
        
        .kpi-trend {
            font-size: 0.9rem;
            margin-top: 0.5rem;
            display: flex;
            align-items: center;
        }
        
        /* Health Alert Banners */
        .alert-banner {
            padding: 1.5rem 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            font-weight: 700;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
            border-left: 8px solid;
            animation: pulse 2s infinite ease-in-out;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0px rgba(255, 255, 255, 0.1); }
            70% { box-shadow: 0 0 0 10px rgba(255, 255, 255, 0); }
            100% { box-shadow: 0 0 0 0px rgba(255, 255, 255, 0); }
        }
        
        .alert-green { background: rgba(46, 204, 113, 0.1); border-color: #2ECC71; color: #2ECC71; }
        .alert-yellow { background: rgba(241, 196, 15, 0.1); border-color: #F1C40F; color: #F1C40F; }
        .alert-red { background: rgba(231, 76, 60, 0.1); border-color: #E74C3C; color: #E74C3C; }
        
        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: transparent;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #1E2530;
            border-radius: 10px 10px 0 0;
            padding: 0 24px;
            color: #8B949E;
            font-weight: 600;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #0072FF !important;
            color: white !important;
            border-bottom: none !important;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 3rem 0;
            color: #4A5568;
            font-size: 0.9rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            margin-top: 4rem;
        }
        
        /* Metric Polish */
        [data-testid="stMetricValue"] {
            font-weight: 700;
            font-size: 2rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

# --- CACHED DATA LOADING ---
@st.cache_data
def load_assets():
    # Use the 25K dataset name found in directory
    data_path = "data/Pump_Predictive_Maintenance_Dataset_25K.xlsx"
    if not os.path.exists(data_path):
        # Fallback to CSV if Excel is missing
        data_path = "data/pump_sensor_data.csv"
        if not os.path.exists(data_path):
            return None, None, None, None, None

    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
        
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    try:
        # Load Phase 2 models
        clf_model = joblib.load("models/failure_model.pkl")
        reg_model = joblib.load("models/rul_model.pkl")
        clf_scaler = joblib.load("models/clf_scaler.pkl")
        reg_scaler = joblib.load("models/reg_scaler.pkl")
        # Feature names are stored in the scaler if it's a newer version or we can get them from the training script
        # Assuming we need to know which features were used:
        # We can extract them from the scaler if it has feature_names_in_
        features = list(clf_scaler.feature_names_in_)
        return df, clf_model, reg_model, clf_scaler, reg_scaler, features
    except Exception as e:
        st.warning(f"Error loading models: {e}")
        return df, None, None, None, None, None

# --- INITIALIZATION ---
apply_custom_styles()
df, clf_model, reg_model, clf_scaler, reg_scaler, features_list = load_assets()

if df is None:
    st.error("❌ Dataset not found. Please run the data generator or pipeline first.")
    st.stop()

# Import engineering logic from new Phase 2 modules
try:
    from src.preprocess import handle_missing_values, handle_categorical_encoding
    from src.features import feature_engineering_pipeline
except ImportError:
    st.error("❌ Modular source components (src.preprocess/src.features) not found.")
    st.stop()

# --- TOP HEADER ---
st.markdown("""
<div class="glass-header">
    <div>
        <h1 class="header-title">PUMP SENTINEL AI</h1>
        <p class="header-subtitle">Predictive Maintenance AI Dashboard | 6-Hour Early Warning</p>
    </div>
    <div style="text-align: right;">
        <span style="color: #636E72; font-size: 0.8rem; text-transform: uppercase; font-weight: 700;">System Status</span><br>
        <span style="color: #2ECC71; font-weight: 700; font-size: 1.1rem;">● OPERATIONAL</span><br>
        <span style="color: #8B949E; font-size: 0.8rem;">Last Updated: """ + datetime.datetime.now().strftime("%H:%M:%S") + """</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown("## 🎛️ CONTROL CENTER")
    st.markdown("---")
    
    pumps = df['Pump_ID'].unique()
    selected_pump = st.selectbox("🏭 Select Industrial Asset", pumps, index=0)
    
    pump_df = df[df['Pump_ID'] == selected_pump].copy().reset_index(drop=True)
    
    # Engineer features first to determine valid simulation range
    # Note: feature_engineering_pipeline might need categorical encoding if not done in preprocess
    processed_pump_df = handle_missing_values(pump_df)
    processed_pump_df = handle_categorical_encoding(processed_pump_df)
    full_engineered = feature_engineering_pipeline(processed_pump_df)
    max_idx = len(full_engineered) - 1
    
    st.markdown("### ⏱️ TIMELINE SIMULATOR")
    
    # Safely find failures in engineered data (Failure_in_Next_6h)
    target_clf = 'Failure_in_Next_6h'
    failures = full_engineered[full_engineered[target_clf] == 1].index
    default_val = failures[0] if len(failures) > 0 else max_idx // 2
    default_val = max(0, min(default_val, max_idx))
    
    time_idx = st.slider("Step Through Operations", 0, max_idx, default_val)
    current_row = full_engineered.iloc[time_idx]
    current_time = current_row['Timestamp']
    st.caption(f"📍 Current View: {current_time.strftime('%Y-%m-%d %H:%M')}")
    
    st.markdown("---")
    st.markdown("### ⚙️ PREFERENCES")
    sensor_options = ['Vibration', 'Bearing_Temperature', 'Inlet_Pressure', 'Outlet_Pressure', 'Flow_Rate', 'Current', 'RPM']
    sensor_options = [s for s in sensor_options if s in full_engineered.columns]
    selected_sensors = st.multiselect("Visible Sensor Streams", sensor_options, default=sensor_options[:3])
    
    conf_thresh = st.slider("🚨 Risk Alert Threshold (%)", 20, 95, 65, step=5)
    
    st.markdown("---")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        show_raw = st.toggle("Show Data", value=False)
    with col_t2:
        auto_ref = st.toggle("Live Sync", value=False)

# --- ENGINE: INFERENCE ---
# Use already engineered row
feature_vals = current_row[features_list].values.reshape(1, -1)

if clf_model and reg_model and clf_scaler:
    scaled_clf_feats = clf_scaler.transform(feature_vals)
    fail_prob = clf_model.predict_proba(scaled_clf_feats)[0][1] * 100
    
    scaled_reg_feats = reg_scaler.transform(feature_vals)
    pred_rul = max(0, float(reg_model.predict(scaled_reg_feats)[0]))
else:
    # Fallback if models are missing
    fail_prob = 15.0
    pred_rul = 450.0

# Alert Selection
if fail_prob >= conf_thresh or pred_rul < 12:
    alert_class = "alert-red"
    alert_icon = "🚨"
    alert_msg = f"CRITICAL RISK: Failure Predicted within {pred_rul:.1f} hours ({fail_prob:.1f}% confidence). Intervene and shutdown immediately."
    status_text = "CRITICAL"
    status_color = "#E74C3C"
elif fail_prob >= (conf_thresh * 0.5) or pred_rul < 72:
    alert_class = "alert-yellow"
    alert_icon = "⚠️"
    alert_msg = f"MAINTENANCE WARNING: Potential degradation detected. RUL is {pred_rul:.1f} hours. Schedule inspection soon."
    status_text = "WARNING"
    status_color = "#F1C40F"
else:
    alert_class = "alert-green"
    alert_icon = "✅"
    alert_msg = "HEALTHY: Pump operating within nominal parameters. No failure risks detected."
    status_text = "HEALTHY"
    status_color = "#2ECC71"

# --- ALERT BANNER ---
st.markdown(f"""
<div class="alert-banner {alert_class}">
    <span style="font-size: 2rem; margin-right: 1.5rem;">{alert_icon}</span>
    <div>{alert_msg}</div>
</div>
""", unsafe_allow_html=True)

# --- KPI CARDS ROW ---
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Health Status</div>
        <div class="kpi-value" style="color: {status_color};">{status_text}</div>
        <div class="kpi-trend" style="color: #8B949E;">Reliability: {100-fail_prob:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    rul_color = "#E74C3C" if pred_rul < 24 else ("#F1C40F" if pred_rul < 168 else "#2ECC71")
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Remaining Useful Life</div>
        <div class="kpi-value" style="color: {rul_color};">{pred_rul:.0f} <small style="font-size: 1rem;">HRS</small></div>
        <div class="kpi-trend">Estimated Ops End</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Failure Probability</div>
        <div class="kpi-value" style="color: {status_color};">{fail_prob:.1f}%</div>
        <div class="kpi-trend">Window: 6 Hours</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    temp = current_row.get('Bearing_Temperature', current_row.get('Temperature_C', 0))
    vib = current_row.get('Vibration', current_row.get('Vibration_mm_s', 0))
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Real-time Telemetry</div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; gap: 10px;">
            <div class="kpi-value kpi-small">{temp:.1f}°C</div>
            <div class="kpi-value kpi-small">{vib:.2f} <small style="font-size: 0.6rem;">mm/s</small></div>
        </div>
        <div class="kpi-trend">Temp / Vibration</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- MAIN TABS ---
t1, t2, t3, t4 = st.tabs(["📈 SENSOR MONITORING", "🔮 FAILURE PREDICTION", "📊 PUMP ANALYTICS", "🗄️ DATA EXPLORER"])

with t1:
    st.markdown("### 📊 Interactive Stream Monitoring")
    # Show last 100 hours
    start_lookback = max(0, time_idx - 100)
    plot_df = full_engineered.iloc[start_lookback:time_idx+1]
    
    fig = go.Figure()
    colors = ['#00C6FF', '#0072FF', '#2ECC71', '#F1C40F', '#E74C3C', '#9B59B6']
    
    for i, sensor in enumerate(selected_sensors):
        fig.add_trace(go.Scatter(
            x=plot_df['Timestamp'],
            y=plot_df[sensor],
            name=sensor.replace('_', ' '),
            line=dict(width=3, color=colors[i % len(colors)]),
            hovertemplate='%{y:.2f} at %{x}<extra></extra>'
        ))
    
    fig.add_vline(x=current_time.timestamp() * 1000, line_dash="dash", line_color="#FFFFFF", annotation_text="Present", annotation_position="top left")
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

with t2:
    st.markdown("### 🔮 Risk Analysis Engine")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        # 6-hour prediction window
        future_idx = min(time_idx + 6, max_idx)
        if future_idx > time_idx:
            forecast_df = full_engineered.iloc[time_idx:future_idx+1].copy()
            # Predict for these
            if clf_model and clf_scaler:
                f_feats = clf_scaler.transform(forecast_df[features_list])
                f_probs = clf_model.predict_proba(f_feats)[:, 1] * 100
            else:
                f_probs = [fail_prob] * len(forecast_df)
            
            fig_bar = px.bar(
                x=forecast_df['Timestamp'],
                y=f_probs.tolist(),
                labels={'x': 'Forecast Horizon', 'y': 'Failure Risk (%)'},
                title="6-Hour Risk Progression",
                color=f_probs.tolist(),
                color_continuous_scale="Reds"
            )
            fig_bar.add_hline(y=conf_thresh, line_dash="dash", line_color="#E74C3C", annotation_text="Danger Zone")
            fig_bar.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with c2:
        st.markdown("#### Probability Spectrum")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = fail_prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': status_color},
                'steps': [
                    {'range': [0, conf_thresh*0.5], 'color': "rgba(46, 204, 113, 0.1)"},
                    {'range': [conf_thresh*0.5, conf_thresh], 'color': "rgba(241, 196, 15, 0.1)"},
                    {'range': [conf_thresh, 100], 'color': "rgba(231, 76, 60, 0.1)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': conf_thresh
                }
            }
        ))
        fig_gauge.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=50, b=20), height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

with t3:
    st.markdown("### 📊 Historical Asset Intel")
    col3_1, col3_2 = st.columns(2)
    
    with col3_1:
        fail_stats = df.groupby('Pump_ID')['Failure_Event'].sum().reset_index()
        fig_stats = px.pie(fail_stats, values='Failure_Event', names='Pump_ID', hole=0.6, title="Failure Distribution Across Fleet", 
                           color_discrete_sequence=px.colors.sequential.RdBu)
        fig_stats.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_stats, use_container_width=True)
        
    with col3_2:
        # Select pump history
        pump_hist = full_engineered[full_engineered['Timestamp'] <= current_time]
        p_fails = pump_hist[pump_hist['Failure_Event'] == 1]
        
        st.markdown(f"#### Incident Log: {selected_pump}")
        if not p_fails.empty:
            st.dataframe(p_fails[['Timestamp', 'Running_hours_total']].tail(), use_container_width=True, hide_index=True)
            uptime = pump_hist['Running_hours_total'].max()
            st.metric("Total Uptime", f"{uptime:,.0f} Hours")
        else:
            st.success("✨ Zero reported failures in historical data for this asset.")

with t4:
    st.markdown("### 🔍 Telemetry Data Explorer")
    if show_raw:
        view_df = full_engineered.iloc[max(0, time_idx-200):time_idx+1].copy()
        st.dataframe(view_df.sort_values("Timestamp", ascending=False), use_container_width=True, height=400)
        
        csv = view_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data Cluster (CSV)",
            data=csv,
            file_name=f"{selected_pump}_export.csv",
            mime="text/csv",
        )
    else:
        st.info("💡 Raw data display is currently locked. Toggle 'Show Data' in the Control Center to unlock.")

# --- FOOTER ---
st.markdown("""
<div class="footer">
    <p><strong>Industrial Predictive Maintenance System v2.1.0</strong></p>
    <p>Engineered with XGBoost AI & Streamlit Industrial Framework • 2026 Sentinel Dynamics</p>
    <p style="opacity: 0.5;">Author: AI Dashboard Architect Placeholder</p>
</div>
""", unsafe_allow_html=True)
