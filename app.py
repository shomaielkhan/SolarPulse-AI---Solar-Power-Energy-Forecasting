import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="SolarPulse AI | Solar Energy Forecast Dashboard", page_icon="☀️", layout="wide")

# Professional UI Styling for SMIU Thesis
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #ff7f0e; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    h1 { color: #1e3d59; text-align: center; font-family: 'Segoe UI', sans-serif; }
    h3 { color: #2c3e50; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Load your trained XGBoost model
    return joblib.load("solar_power_xgb_model.pkl")

try:
    model = load_assets()
    expected_features = model.get_booster().feature_names
except Exception as e:
    st.error(f"⚠️ Model Load Error: {e}")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3222/3222672.png", width=80)
    st.title("Solar Energy Generation Dashboard")
    uploaded_file = st.file_uploader("Upload CSV Data", type="csv")
    
    st.markdown("---")
    st.subheader("Model Performance")
    st.metric("R² Score", "0.976")
    st.metric("MAE", "7.44 kW")
    
    app_mode = st.radio("Select Dashboard:", ["72-Hour Forecast", "Model Evaluation (EDA)"])

# --- MAIN CONTENT ---
if uploaded_file:
    try:
        # 1. Load data
        df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True).sort_index()
        
        # 2. Timing Logic
        time_diff = df.index[1] - df.index[0]
        points_per_day = int(pd.Timedelta(days=1) / time_diff)
        
        # Predict on a slice for analysis
        df_eval = df.head(points_per_day * 7).copy() 
        df_eval['Predicted_Power'] = np.maximum(0, model.predict(df_eval[expected_features]))

        if app_mode == "72-Hour Forecast":
            st.title("📅 Project Goal: 72-Hour Solar Forecast")
            
            # Slice for 3 days (72 hours)
            df_3d = df_eval.head(points_per_day * 3)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Peak Power (3-Day)", f"{df_3d['Predicted_Power'].max():.2f} kW")
            m2.metric("Total Energy (kWh)", f"{df_3d['Predicted_Power'].sum() * (time_diff.seconds/3600):.1f}")
            m3.metric("Status", "Operational")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_3d.index, y=df_3d['Predicted_Power'], 
                                     mode='lines', name='72-Hour Prediction',
                                     line=dict(color='#ff7f0e', width=3),
                                     fill='tozeroy', fillcolor='rgba(255, 127, 14, 0.1)'))
            
            fig.update_layout(template="plotly_white", height=500, 
                              xaxis_title="Time Horizon (72 Hours)", 
                              yaxis_title="DC Power (kW)",
                              hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Successfully generated a 72-hour forecast for {len(df_3d)} intervals.")

        else:
            # --- UPDATED EDA TAB ---
            st.title("📊 Model Validation & Performance Analysis")

            # 1. 24-HOUR HORIZON ACTUAL VS PREDICTED
            st.subheader("🎯 24-Hour Horizon: Actual vs. Predicted (Day-Ahead Accuracy)")
            df_24h = df_eval.head(points_per_day) # Exactly 24 hours
            
            fig_24h = go.Figure()
            fig_24h.add_trace(go.Scatter(x=df_24h.index, y=df_24h['power_dc'], 
                                         name='Actual Sensor Data', 
                                         line=dict(color='#1e3d59', width=2)))
            fig_24h.add_trace(go.Scatter(x=df_24h.index, y=df_24h['Predicted_Power'], 
                                         name='AI Prediction', 
                                         line=dict(color='#ff7f0e', width=2, dash='dot')))
            
            fig_24h.update_layout(template="plotly_white", height=450, 
                                  xaxis_title="24 Hour Timeline", 
                                  yaxis_title="DC Power (kW)", 
                                  hovermode="x unified")
            st.plotly_chart(fig_24h, use_container_width=True)

            # 2. PREDICTION CONSISTENCY (SCATTER PLOT)
            st.subheader("⚖️ Prediction Consistency: Actual vs. Predicted")
            # Using data points to show consistency trend
            fig_scatter = px.scatter(df_eval, x='power_dc', y='Predicted_Power', 
                                     labels={'power_dc': 'Actual Power (kW)', 'Predicted_Power': 'Predicted Power (kW)'},
                                     opacity=0.5, color_discrete_sequence=['#1e3d59'])
            
            # Add the "Ideal" 45-degree line for reference
            max_val = max(df_eval['power_dc'].max(), df_eval['Predicted_Power'].max())
            fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, 
                                  line=dict(color="Red", dash="dash"))
            
            fig_scatter.update_layout(template="plotly_white", height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)

            # 3. FEATURE IMPORTANCE
            st.subheader("🧠 Model Intelligence: Top 5 Drivers")
            importances = pd.Series(model.feature_importances_, index=expected_features).sort_values(ascending=True).tail(5)
            fig_imp = px.bar(importances, orientation='h', color_discrete_sequence=['#ff7f0e'])
            fig_imp.update_layout(template="plotly_white", height=400, 
                                  xaxis_title="Importance Score", 
                                  yaxis_title="Weather Features")
            st.plotly_chart(fig_imp, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Processing Error: {e}")
else:
    st.info("👋 System Ready. Please upload your CSV data to generate the analysis.")