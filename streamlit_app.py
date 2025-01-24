import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import io

# Function to load and preprocess data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Function to calculate performance score
def calculate_performance_score(df):
    features = ['delivery_time', 'quality_score', 'cost_efficiency', 'communication_rating', 'vendor_performance']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    
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

# Example datasets
example_data = {
    "IT Service Providers": """vendor_name,delivery_time,quality_score,cost_efficiency,communication_rating,vendor_performance,historical_performance
TechPro Solutions,8,9.2,85,8.7,88,89
GlobalIT Services,7,8.5,92,9.1,90,91
InnovateIT Consulting,9,9.5,78,9.3,87,88
FastTrack Systems,6,8.8,88,8.9,89,90
QualityFirst IT,10,9.7,72,8.5,85,86""",

    "Manufacturing Suppliers": """vendor_name,delivery_time,quality_score,cost_efficiency,communication_rating,vendor_performance,historical_performance
PrecisionParts Inc.,5,9.5,88,9.0,92,93
GlobalGear Manufacturing,7,9.8,82,9.2,90,91
QuickAssembly Solutions,4,8.7,95,8.8,89,90
IndustrialTech Suppliers,8,9.9,75,9.5,91,92
EcoFriendly Components,6,9.6,85,9.1,93,94""",

    "Logistics Partners": """vendor_name,delivery_time,quality_score,cost_efficiency,communication_rating,vendor_performance,historical_performance
SpeedyShip Express,5,8.9,92,8.7,88,89
GlobalFreight Solutions,7,9.3,85,9.0,89,90
QuickDeliver Co.,4,8.8,94,8.9,90,91
EcoLogistics Services,6,9.1,88,9.2,91,92
TechTrack Shipping,8,9.5,80,9.4,92,93"""
}

# Streamlit app
st.title("AI-Powered Vendor Performance Analysis Tool")

# Data selection section
st.header("Select Data Source")
data_source = st.radio("Choose your data source:", ("Demo Data", "Upload Your Own Data"))

if data_source == "Demo Data":
    demo_option = st.selectbox("Choose a demo dataset:", list(example_data.keys()))
    if demo_option:
        data = pd.read_csv(io.StringIO(example_data[demo_option]))
        st.write(f"Using demo data: {demo_option}")
else:
    uploaded_file = st.file_uploader("Upload your vendor data CSV", type="csv")
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Your data has been uploaded successfully.")
    else:
        st.warning("Please upload a CSV file to proceed.")
        data = None

# Run Analysis button
run_analysis = st.button("Run Analysis")

if run_analysis and data is not None:
    st.header("Analysis Results")
    
    st.subheader("Vendor Data Overview")
    st.write(data.head())

    performance_score = calculate_performance_score(data)
    data['Performance Score'] = performance_score

    future_performance = predict_future_performance(data, performance_score)
    data['Predicted Future Performance'] = future_performance

    st.subheader("Vendor Performance Analysis")
    fig = px.scatter(data, x='Performance Score', y='Predicted Future Performance', 
                     hover_data=['vendor_name'], color='historical_performance')
    st.plotly_chart(fig)

    st.subheader("Top Performing Vendors")
    top_vendors = data.nlargest(5, 'Performance Score')[['vendor_name', 'Performance Score', 'Predicted Future Performance']]
    st.write(top_vendors)

    st.subheader("Vendors Needing Improvement")
    bottom_vendors = data.nsmallest(5, 'Performance Score')[['vendor_name', 'Performance Score', 'Predicted Future Performance']]
    st.write(bottom_vendors)

    st.subheader("Download Full Analysis Report")
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="vendor_analysis_report.csv",
        mime="text/csv",
    )

    # Explanation section
    st.header("How to Interpret the Scores")
    st.markdown("""
    ### Performance Score
    - Range: Typically between -2 to 2 (standardized scale)
    - Interpretation: 
      - Higher scores indicate better overall performance
      - Scores above 0 are above average
      - Scores below 0 are below average

    ### Predicted Future Performance
    - Range: Similar to Performance Score
    - Interpretation:
      - Higher values suggest expected improvement in future performance
      - Lower values indicate potential decline in performance

    ### Metrics Explained
    1. **Delivery Time**: Lower is better (faster delivery)
    2. **Quality Score**: Higher is better (0-10 scale)
    3. **Cost Efficiency**: Higher is better (0-100 scale)
    4. **Communication Rating**: Higher is better (0-10 scale)
    5. **Vendor Performance**: Overall performance rating (0-100 scale)
    6. **Historical Performance**: Past performance metric (0-100 scale)

    ### Using This Information
    - Identify top performers for potential increased business or learning best practices
    - Focus on vendors needing improvement for targeted support or reevaluation
    - Use predicted future performance for strategic planning and risk management
    """)
elif run_analysis and data is None:
    st.error("Please select a demo dataset or upload your own data before running the analysis.")
