import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from train import clean_data

# --- Config ---
st.set_page_config(page_title="Logistics Delay Dashboard", layout="wide")

# --- Data & Model Loaders ---
@st.cache_resource
def load_models():
    models = {}
    for name in ['catboost', 'lightgbm', 'xgboost']:
        path = f"models/{name}_model.pkl"
        if os.path.exists(path):
            models[name.capitalize()] = joblib.load(path)
    return models

@st.cache_data
def load_sample_data():
    if os.path.exists("data/X_test.parquet") and os.path.exists("data/y_test.parquet"):
        X = pd.read_parquet("data/X_test.parquet")
        y = pd.read_parquet("data/y_test.parquet")
        df = X.copy()
        df['IS_DELAYED_ACTUAL'] = y['IS_DELAYED']
        return df
    elif os.path.exists("data/raw_shipments.parquet"):
        df = pd.read_parquet("data/raw_shipments.parquet")
        return clean_data(df.head(5000)) # Sample for app rendering speed
    else:
        return pd.DataFrame()

# --- App UI ---
st.title("📦 Logistics Delay Prediction Dashboard")
st.markdown("Predictive analytics for shipment delays. Adjust interventions dynamically.")

models = load_models()
if not models:
    st.warning("No trained models found! Please run train.py first.")
    st.stop()

df = load_sample_data()
if df.empty:
    st.warning("No data found! Please prepare data with data_ingestion.py and train.py.")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")
selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
selected_model = models[selected_model_name]

# Dynamic threshold slider to tune recall
threshold = st.sidebar.slider(
    "Probability Threshold for Delay", 
    min_value=0.01, 
    max_value=0.99, 
    value=0.30, 
    step=0.01,
    help="Lower threshold = Higher Recall (Identify more potential delays, risk of false positives)."
)

# --- Perform Inference ---
# Drop ground truth columns if present
X_infer = df.drop(columns=['IS_DELAYED_ACTUAL', 'IS_DELAYED'], errors='ignore')

if os.path.exists("models/cat_features.pkl"):
    cat_features = joblib.load("models/cat_features.pkl")
else:
    cat_features = []

# Prepare data types for LightGBM/XGBoost
if selected_model_name in ['LightGBM', 'XGBoost']:
    for col in cat_features:
        if col in X_infer.columns:
            X_infer[col] = X_infer[col].astype('category')

with st.spinner("Generating predictions..."):
    # Probabilities for class 1
    probabilities = selected_model.predict_proba(X_infer)[:, 1]
    
df['DELAY_PROBABILITY'] = probabilities
# Output prediction based on slider
df['PREDICTED_DELAY'] = (df['DELAY_PROBABILITY'] >= threshold).astype(int)

# --- Executive Top Level Metrics ---
col1, col2, col3 = st.columns(3)
total_shipments = len(df)
predicted_delays = df['PREDICTED_DELAY'].sum()
delay_rate = (predicted_delays / total_shipments) * 100 if total_shipments else 0

col1.metric("Total Shipments Context", f"{total_shipments:,}")
col2.metric("Predicted Delays (Thresholded)", f"{predicted_delays:,}")
col3.metric("Predicted Delay Rate", f"{delay_rate:.1f}%")

st.markdown("---")

# --- Visualizations ---
col_v1, col_v2 = st.columns(2)

with col_v1:
    st.subheader("Delay Rate by Delivery Partner Carrier")
    if 'DELIVERY_PARTNER_CARRIER' in df.columns:
        partner_delays = df.groupby('DELIVERY_PARTNER_CARRIER')['PREDICTED_DELAY'].mean().reset_index()
        partner_delays['PREDICTED_DELAY'] *= 100 # percentage
        partner_delays = partner_delays.sort_values('PREDICTED_DELAY', ascending=False).head(15)
        
        fig = px.bar(
            partner_delays, 
            y='DELIVERY_PARTNER_CARRIER', 
            x='PREDICTED_DELAY',
            orientation='h',
            labels={'PREDICTED_DELAY': 'Delay Rate (%)', 'DELIVERY_PARTNER_CARRIER': 'Carrier'},
            color='PREDICTED_DELAY',
            color_continuous_scale='Reds'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("DELIVERY_PARTNER_CARRIER column not found.")

with col_v2:
    st.subheader("Delay Rate by Destination Country")
    if 'DESTINATION_COUNTRY' in df.columns:
        country_delays = df.groupby('DESTINATION_COUNTRY')['PREDICTED_DELAY'].mean().reset_index()
        country_delays['PREDICTED_DELAY'] *= 100
        country_delays = country_delays.sort_values('PREDICTED_DELAY', ascending=False).head(15)
        
        fig = px.bar(
            country_delays, 
            y='DESTINATION_COUNTRY', 
            x='PREDICTED_DELAY',
            orientation='h',
            labels={'PREDICTED_DELAY': 'Delay Rate (%)', 'DESTINATION_COUNTRY': 'Country'},
            color='PREDICTED_DELAY',
            color_continuous_scale='Oranges'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("DESTINATION_COUNTRY column not found.")

# --- Feature Importance Chart ---
st.subheader("Global Feature Importance")
if selected_model_name == 'CatBoost':
    importances = selected_model.get_feature_importance()
elif hasattr(selected_model, 'feature_importances_'):
    importances = selected_model.feature_importances_
else:
    importances = None

if importances is not None:
    df_imp = pd.DataFrame({
        'Feature': X_infer.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(20)
    
    fig_imp = px.bar(
        df_imp,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.info("Feature importance not natively mapped for this model object.")

# --- Actionable Data Table ---
st.subheader("Actionable Shipments (Predicted Delayed)")
st.caption("These shipments meet or exceed the selected threshold probability. Prioritize mitigating these first.")

# Filter only predicted delayed and sort by highest risk
delayed_df = df[df['PREDICTED_DELAY'] == 1].sort_values('DELAY_PROBABILITY', ascending=False)

# Display a cleaner subset of columns to the stakeholder if possible
display_cols = [
    'DELAY_PROBABILITY', 'X_3PL_NAME', 'DELIVERY_PARTNER_CARRIER', 
    'DESTINATION_COUNTRY', 'ROUTE_ID', 'LABEL_PRINT_YEAR_WEEK_UTC'
]
available_cols = [c for c in display_cols if c in delayed_df.columns]
all_other_cols = [c for c in delayed_df.columns if c not in available_cols and c != 'PREDICTED_DELAY']

final_display = delayed_df[available_cols + all_other_cols]

st.dataframe(
    final_display.head(1000), # display top 1000 for interface performance
    use_container_width=True,
    column_config={
        "DELAY_PROBABILITY": st.column_config.ProgressColumn(
            "Delay Probability",
            help="Probability the shipment will be delayed",
            format="%.2f",
            min_value=0,
            max_value=1,
        ),
        "IS_DELAYED_ACTUAL": "Actual Delay (0/1)"
    }
)
