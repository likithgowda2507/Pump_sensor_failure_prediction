import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Pump Maintenance Dashboard", page_icon="⚙️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Global styles and metrics */
    .metric-card {
        background-color: #1e1e2d;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
        margin-bottom: 24px;
        border: 1px solid #33334c;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 12px 0;
        font-family: 'Inter', sans-serif;
    }
    .metric-label {
        color: #a0a0b0;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 500;
    }
    
    /* Status Colors */
    .status-green { color: #00e676; text-shadow: 0 0 10px rgba(0,230,118,0.3); }
    .status-yellow { color: #ffeb3b; text-shadow: 0 0 10px rgba(255,235,59,0.3); }
    .status-red { color: #ff1744; text-shadow: 0 0 10px rgba(255,23,68,0.3); }
    .status-blue { color: #4fc3f7; }
    
    /* Header Container */
    .header-container {
        padding: 20px 0 40px 0;
        text-align: center;
    }
    .header-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #4fc3f7, #00e676);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .header-subtitle {
        color: #a0a0b0;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHE DATA & MODEL ---
@st.cache_data
def load_data():
    df = pd.read_csv("pump_sensor_data.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by=['Pump_ID', 'Timestamp'])
    
    # Forward fill missing values (if any)
    sensor_cols = ['Temperature_C', 'Vibration_mm_s', 'Pressure_bar', 'Flow_rate_lpm', 
                   'Motor_current_A', 'Voltage_V', 'Power_kW', 'Efficiency_percent', 
                   'Noise_level_dB', 'Oil_level_percent']
    df[sensor_cols] = df.groupby('Pump_ID')[sensor_cols].ffill()
    df[sensor_cols] = df[sensor_cols].bfill() # Fallback
    
    return df

@st.cache_resource
def train_model(df):
    """
    Train a lightweight Random Forest model to predict probability of Failure in the next 6 hours.
    Uses the entire dataset for demonstration purposes.
    """
    features = ['Temperature_C', 'Vibration_mm_s', 'Pressure_bar', 'Flow_rate_lpm', 
                'Motor_current_A', 'Voltage_V', 'Power_kW', 'Efficiency_percent', 
                'Noise_level_dB', 'Oil_level_percent', 'Running_hours_total']
    target = 'Failure_in_6h'
    
    X = df[features]
    y = df[target]
    
    # Subsample if dataset is very large to speed up initial load
    if len(df) > 50000:
        sample_idx = np.random.choice(len(df), 50000, replace=False)
        X = X.iloc[sample_idx]
        y = y.iloc[sample_idx]
        
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, features

# --- MAIN APP ROUTINE ---
def main():
    st.markdown("""
        <div class="header-container">
            <div class="header-title">Pump Sentinel Analytics</div>
            <div class="header-subtitle">Real-time Predictive Maintenance & Health Monitoring</div>
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading telemetry data and predictive models..."):
        df = load_data()
        model, features = train_model(df)

    # --- SIDEBAR CONTROLS ---
    st.sidebar.title("⚙️ Controls")
    pump_ids = df['Pump_ID'].unique()
    selected_pump = st.sidebar.selectbox("Select Target Equipment", pump_ids)
    
    pump_df = df[df['Pump_ID'] == selected_pump].copy()
    pump_df = pump_df.reset_index(drop=True)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Time Simulation Explorer")
    st.sidebar.caption("Move slider to simulate time progression and view predictions at a specific point.")
    
    max_idx = len(pump_df) - 1
    default_idx = max_idx - 168  # View exactly 1 week prior to the end for good trend display
    if default_idx < 0: default_idx = max_idx
    
    current_idx = st.sidebar.slider("Timeline Explorer", 0, max_idx, default_idx)
    
    # Get the current record
    current_record = pump_df.iloc[current_idx]
    current_time = current_record['Timestamp']
    
    history_df = pump_df.iloc[max(0, current_idx - 168) : current_idx + 1]
    
    st.sidebar.markdown(f"**Current Timestamp:** `{current_time.strftime('%Y-%m-%d %H:%M:%S')}`")

    # --- PREDICTIONS & LOGIC ---
    current_features = current_record[features].values.reshape(1, -1)
    
    # Use the model to predict probability of a failure in the 6H window
    failure_prob = model.predict_proba(current_features)[0][1] * 100
    
    predicted_rul = current_record['RUL_hours']

    # Health Alert Logic
    if failure_prob > 50 or predicted_rul < 24:
        alert_status = "CRITICAL"
        status_color = "status-red"
    elif failure_prob > 15 or predicted_rul < 168:
        alert_status = "WARNING"
        status_color = "status-yellow"
    else:
        alert_status = "HEALTHY"
        status_color = "status-green"

    # --- KPI DISPLAY ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Health Status</div>
            <div class="metric-value {status_color}">{alert_status}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        rul_color = "status-green" if predicted_rul > 168 else ("status-yellow" if predicted_rul > 24 else "status-red")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted RUL</div>
            <div class="metric-value {rul_color}">{predicted_rul:.0f}h</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        prob_color = "status-green" if failure_prob <= 15 else ("status-red" if failure_prob > 50 else "status-yellow")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Failure Probability (6H)</div>
            <div class="metric-value {prob_color}">{failure_prob:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Running Hours</div>
            <div class="metric-value status-blue">{current_record['Running_hours_total']:.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    # --- SENSOR TRENDS ---
    st.markdown("### 📊 Sensor Telemetry (Past 7 Days)")
    tab1, tab2, tab3 = st.tabs(["🌡️ Temp & High-Freq", "💨 Pressure & Flow", "⚡ Power Diagnostics"])
    
    # Plotly standard layout settings
    layout_settings = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    
    with tab1:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_temp = px.line(history_df, x='Timestamp', y='Temperature_C', title='Temperature (°C)')
            fig_temp.update_traces(line_color='#ff8a65', line_width=2)
            fig_temp.update_layout(**layout_settings)
            st.plotly_chart(fig_temp, use_container_width=True)
        with col_t2:
            fig_vib = px.line(history_df, x='Timestamp', y='Vibration_mm_s', title='Vibration (mm/s)')
            fig_vib.update_traces(line_color='#ba68c8', line_width=2)
            fig_vib.update_layout(**layout_settings)
            st.plotly_chart(fig_vib, use_container_width=True)
            
    with tab2:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_press = px.line(history_df, x='Timestamp', y='Pressure_bar', title='Pressure (bar)')
            fig_press.update_traces(line_color='#4dd0e1', line_width=2)
            fig_press.update_layout(**layout_settings)
            st.plotly_chart(fig_press, use_container_width=True)
        with col_p2:
            fig_flow = px.line(history_df, x='Timestamp', y='Flow_rate_lpm', title='Flow Rate (lpm)')
            fig_flow.update_traces(line_color='#81c784', line_width=2)
            fig_flow.update_layout(**layout_settings)
            st.plotly_chart(fig_flow, use_container_width=True)
            
    with tab3:
        col_pw1, col_pw2 = st.columns(2)
        with col_pw1:
            fig_power = px.line(history_df, x='Timestamp', y='Power_kW', title='Power Consumption (kW)')
            fig_power.update_traces(line_color='#ffd54f', line_width=2)
            fig_power.update_layout(**layout_settings)
            st.plotly_chart(fig_power, use_container_width=True)
        with col_pw2:
            fig_eff = px.line(history_df, x='Timestamp', y='Efficiency_percent', title='Efficiency (%)')
            fig_eff.update_traces(line_color='#4db6ac', line_width=2)
            fig_eff.update_layout(**layout_settings)
            st.plotly_chart(fig_eff, use_container_width=True)

    # --- PREDICTION WINDOW & HISTORY ---
    st.markdown("---")
    col_bottom1, col_bottom2 = st.columns([1.5, 1])
    
    with col_bottom1:
        st.markdown("### 🔮 6-Hour Prediction Window")
        
        future_idx_end = min(max_idx, current_idx + 6)
        future_df = pump_df.iloc[current_idx : future_idx_end + 1].copy()
        
        if len(future_df) > 1:
            future_probs = model.predict_proba(future_df[features])[:, 1] * 100
            future_df['Predicted_Failure_Prob'] = future_probs
            
            fig_future = go.Figure()
            # Dynamic color gradient based on probability
            colors = ['#ff1744' if p > 50 else ('#ffeb3b' if p > 15 else '#00e676') for p in future_probs]
            
            fig_future.add_trace(go.Bar(
                x=future_df['Timestamp'], 
                y=future_df['Predicted_Failure_Prob'],
                marker_color=colors,
                name='Failure Probability (%)',
                text=[f"{p:.1f}%" for p in future_probs],
                textposition='auto'
            ))
            fig_future.add_hline(y=50, line_dash="dash", line_color="rgba(255,23,68,0.8)", annotation_text="Critical Threshold (50%)", annotation_position="top left")
            fig_future.update_layout(
                yaxis_title="Probability (%)",
                yaxis_range=[0, 100],
                xaxis_title="Future Timestamp",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_future, use_container_width=True)
        else:
            st.info("Insufficient future data available for the 6-hour prediction window at this point in the timeline.")
            
    with col_bottom2:
        st.markdown("### 📝 Historical Failures")
        failures = pump_df[(pump_df['Failure_event'] == 1) & (pump_df['Timestamp'] <= current_time)]
        
        if len(failures) > 0:
            hist_disp = failures[['Timestamp', 'Failure_type', 'Running_hours_total']].copy()
            hist_disp['Timestamp'] = hist_disp['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            hist_disp.columns = ['Failure Time', 'Type', 'Run Hours']
            
            st.dataframe(
                hist_disp.sort_values(by='Failure Time', ascending=False), 
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.success("✅ No historical failures recorded for this equipment up to the current simulated time.")

if __name__ == "__main__":
    main()
