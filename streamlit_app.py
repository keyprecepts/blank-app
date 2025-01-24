import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# Function to load and preprocess data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Function to calculate performance score
def calculate_performance_score(df):
    features = ['delivery_time', 'quality_score', 'cost_efficiency', 'communication_rating']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    # Use Random Forest for importance-based scoring
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(scaled_features, df['historical_performance'])
    
    importance = rf.feature_importances_
    performance_score = np.dot(scaled_features, importance)
    return performance_score

# Function to predict future performance
def predict_future_performance(df, performance_score):
    X = df[['historical_performance']].values
    y = performance_score
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    future_performance = model.predict(X)
    return future_performance

# Streamlit app
st.title("AI-Powered Vendor Performance Analysis Tool")

uploaded_file = st.file_uploader("Upload vendor data CSV", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    st.subheader("Vendor Data Overview")
    st.write(df.head())
    
    performance_score = calculate_performance_score(df)
    df['Performance Score'] = performance_score
    
    future_performance = predict_future_performance(df, performance_score)
    df['Predicted Future Performance'] = future_performance
    
    st.subheader("Vendor Performance Analysis")
    fig = px.scatter(df, x='Performance Score', y='Predicted Future Performance', 
                     hover_data=['vendor_name'], color='historical_performance')
    st.plotly_chart(fig)
    
    st.subheader("Top Performing Vendors")
    top_vendors = df.nlargest(5, 'Performance Score')[['vendor_name', 'Performance Score', 'Predicted Future Performance']]
    st.write(top_vendors)
    
    st.subheader("Vendors Needing Improvement")
    bottom_vendors = df.nsmallest(5, 'Performance Score')[['vendor_name', 'Performance Score', 'Predicted Future Performance']]
    st.write(bottom_vendors)
    
    st.subheader("Download Full Analysis Report")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="vendor_analysis_report.csv",
        mime="text/csv",
    )