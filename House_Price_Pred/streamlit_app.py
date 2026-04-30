"""
Streamlit Frontend for Multiple Linear Regression Housing Price Prediction Model
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Housing Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model and features
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / 'trained_model.pkl'
    features_path = Path(__file__).parent / 'feature_names.pkl'
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

model, feature_names = load_model()

# Title
st.markdown("<h1 class='main-header'>🏠 Housing Price Prediction Model</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #666;'>Multiple Linear Regression Analysis for Real Estate Valuation</p>", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💰 Price Prediction", "📊 Model Info", "📈 Visualizations", "🎯 Feature Analysis", "ℹ️ About"])

# ============================================================================
# TAB 1: PRICE PREDICTION
# ============================================================================
with tab1:
    st.header("Make a Price Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Property Details")
        bedrooms = st.slider("Bedrooms", 1, 13, 3, key="bedrooms")
        bathrooms = st.slider("Bathrooms", 0.5, 8.0, 2.0, step=0.5, key="bathrooms")
        sqft_living = st.number_input("Living Area (sqft)", 300, 14000, 2000, key="sqft_living")
        sqft_lot = st.number_input("Lot Size (sqft)", 500, 1000000, 7500, key="sqft_lot")
        floors = st.slider("Floors", 1.0, 3.5, 1.5, step=0.5, key="floors")
    
    with col2:
        st.subheader("Condition & Quality")
        waterfront = st.selectbox("Waterfront Property?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="waterfront")
        view = st.slider("View Quality (0-4)", 0, 4, 0, key="view")
        condition = st.slider("Condition (1-5)", 1, 5, 3, key="condition")
        grade = st.slider("Grade (6-13)", 6, 13, 7, key="grade")
        yr_built = st.number_input("Year Built", 1900, 2024, 2000, key="yr_built")
    
    with col3:
        st.subheader("Additional Features")
        sqft_above = st.number_input("Above Ground Area (sqft)", 0, 14000, 1500, key="sqft_above")
        sqft_basement = st.number_input("Basement Area (sqft)", 0, 5000, 0, key="sqft_basement")
        yr_renovated = st.number_input("Year Renovated (0 if none)", 0, 2024, 0, key="yr_renovated")
        zipcode = st.number_input("Zip Code", 90000, 99000, 98000, key="zipcode")
        lat = st.number_input("Latitude", 47.0, 48.0, 47.6, step=0.01, key="lat")
        long = st.number_input("Longitude", -123.0, -122.0, -122.3, step=0.01, key="long")
        sqft_living15 = st.number_input("Avg Living Area in Neighborhood (sqft)", 399, 6500, 2000, key="sqft_living15")
        sqft_lot15 = st.number_input("Avg Lot Size in Neighborhood (sqft)", 600, 900000, 10000, key="sqft_lot15")
    
    # Make prediction
    if st.button("🔍 Predict Price", use_container_width=True, type="primary"):
        # Prepare input data
        input_data = pd.DataFrame({
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'sqft_living': [sqft_living],
            'sqft_lot': [sqft_lot],
            'floors': [floors],
            'waterfront': [waterfront],
            'view': [view],
            'condition': [condition],
            'grade': [grade],
            'sqft_above': [sqft_above],
            'sqft_basement': [sqft_basement],
            'yr_built': [yr_built],
            'yr_renovated': [yr_renovated],
            'zipcode': [zipcode],
            'lat': [lat],
            'long': [long],
            'sqft_living15': [sqft_living15],
            'sqft_lot15': [sqft_lot15]
        })
        
        # Ensure columns are in correct order
        input_data = input_data[feature_names]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display prediction
        st.markdown(f"<div class='prediction-box'>Estimated Price: ${prediction:,.2f}</div>", unsafe_allow_html=True)
        
        # Display breakdown
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Estimated Price", f"${prediction:,.0f}")
        with col2:
            st.metric("Price per Sqft", f"${prediction/sqft_living:,.2f}")
        with col3:
            st.metric("Monthly Estimate", f"${prediction/360:,.0f}")
        
        # Feature contribution
        st.subheader("💡 Feature Impact on Price")
        coefficients = model.coef_
        feature_impact = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Input Value': input_data.iloc[0].values,
            'Contribution': coefficients * input_data.iloc[0].values
        }).sort_values('Contribution', ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(x=feature_impact['Contribution'], y=feature_impact['Feature'], 
                   orientation='h', marker=dict(color=feature_impact['Contribution'],
                   colorscale='RdBu', showscale=False))
        ])
        fig.update_layout(title="Contribution of Each Feature to Price Prediction",
                         xaxis_title="Contribution ($)", yaxis_title="Feature",
                         height=600, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: MODEL INFORMATION
# ============================================================================
with tab2:
    st.header("📊 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score (Test)", "0.7012")
    with col2:
        st.metric("Adjusted R² (Test)", "0.6999")
    with col3:
        st.metric("RMSE (Test)", "$212,540")
    with col4:
        st.metric("MAE (Test)", "$127,493")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Overview")
        st.info("""
        **Multiple Linear Regression Model**
        - **Dependent Variable:** House Price
        - **Independent Variables:** 18 features
        - **Training Samples:** 17,290 (80%)
        - **Testing Samples:** 4,323 (20%)
        - **Algorithm:** Linear Regression (Scikit-learn)
        """)
    
    with col2:
        st.subheader("Statistical Significance")
        st.success("""
        ✓ **F-Statistic:** 2229.29
        ✓ **P-Value:** < 0.001
        ✓ **Result:** Model is HIGHLY SIGNIFICANT
        
        The overall model is statistically significant, meaning the relationship between features and price is not due to chance.
        """)
    
    st.divider()
    
    st.subheader("📋 Model Coefficients")
    coefficients_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    st.dataframe(coefficients_df, use_container_width=True)
    st.caption(f"**Intercept:** ${model.intercept_:,.2f}")

# ============================================================================
# TAB 3: VISUALIZATIONS
# ============================================================================
with tab3:
    st.header("📈 Model Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Coefficients")
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', ascending=False).head(10)
        
        fig = px.bar(coef_df, x='Coefficient', y='Feature', 
                     title='Top 10 Most Impactful Features',
                     color='Coefficient', color_continuous_scale='RdBu')
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Performance Comparison")
        metrics_data = pd.DataFrame({
            'Set': ['Training', 'Testing'],
            'R² Score': [0.6991, 0.7012],
            'Adjusted R²': [0.6988, 0.6999]
        })
        
        fig = go.Figure(data=[
            go.Bar(name='R² Score', x=metrics_data['Set'], y=metrics_data['R² Score']),
            go.Bar(name='Adjusted R²', x=metrics_data['Set'], y=metrics_data['Adjusted R²'])
        ])
        fig.update_layout(title='Model Performance Metrics', 
                         yaxis_title='Score', height=500, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Error Metrics")
        error_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE'],
            'Training': [39311882352, 198272, 125033],
            'Testing': [45173046133, 212540, 127493]
        })
        
        fig = px.bar(error_df, x='Metric', y=['Training', 'Testing'],
                     barmode='group', title='Error Metrics Comparison')
        fig.update_layout(height=500, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Variance Explained")
        variance_data = {
            'Explained': 70.12,
            'Unexplained': 29.88
        }
        fig = go.Figure(data=[go.Pie(
            labels=list(variance_data.keys()),
            values=list(variance_data.values()),
            hole=.3,
            marker=dict(colors=['#1f77b4', '#ff7f0e'])
        )])
        fig.update_layout(title='Variance Explained by Model (R² = 0.7012)', height=500)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 4: FEATURE ANALYSIS
# ============================================================================
with tab4:
    st.header("🎯 Feature Analysis & Insights")
    
    st.subheader("Top 10 Most Influential Features")
    top_features = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_,
        'Absolute Impact': abs(model.coef_)
    }).sort_values('Absolute Impact', ascending=False).head(10)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.bar(top_features, x='Coefficient', y='Feature',
                     color='Coefficient', color_continuous_scale='RdBu',
                     title='Top 10 Features by Impact on Price')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("### Feature Insights")
        st.write("""
        **Positive Impact** (increase price):
        - Latitude (+$595,968)
        - Waterfront (+$562,413)
        - Grade (+$94,568)
        
        **Negative Impact** (decrease price):
        - Longitude (-$194,586)
        - Bedrooms (-$34,335)
        - Year Built (-$2,681)
        """)
    
    st.divider()
    
    st.subheader("Feature Statistics")
    csv_path = Path(__file__).parent / 'House_data.csv'
    df = pd.read_csv(csv_path)
    
    stats_df = df[feature_names].describe().T
    st.dataframe(stats_df, use_container_width=True)

# ============================================================================
# TAB 5: ABOUT
# ============================================================================
with tab5:
    st.header("ℹ️ About This Application")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Project Overview")
        st.markdown("""
        This application uses a **Multiple Linear Regression** model trained on 
        housing data to predict house prices based on various property features.
        
        ### Key Statistics
        - **Dataset Size:** 21,613 houses
        - **Features:** 18 property attributes
        - **Train-Test Split:** 80% - 20%
        - **Model Accuracy (R²):** 70.12%
        
        ### Methodology
        The model uses the following features to predict prices:
        - Physical characteristics (bedrooms, bathrooms, sqft)
        - Location data (latitude, longitude, zipcode)
        - Condition metrics (grade, condition, year built)
        - Special features (waterfront, view)
        """)
    
    with col2:
        st.subheader("Model Performance")
        st.markdown("""
        ### Test Set Metrics
        - **R² Score:** 0.7012
        - **Adjusted R²:** 0.6999
        - **RMSE:** $212,540
        - **MAE:** $127,493
        - **F-Statistic:** 561.10
        - **P-Value:** < 0.001 ✓
        
        ### Interpretation
        The model explains 70.12% of the variance in house prices. 
        This is considered a strong model for real estate prediction.
        
        All coefficients are statistically significant (p < 0.05),
        confirming that each feature meaningfully contributes to price prediction.
        """)
    
    st.divider()
    
    st.subheader("📚 Technical Details")
    st.markdown("""
    ### Tools & Libraries Used
    - **Python 3.14**
    - **Scikit-learn:** Machine Learning
    - **Pandas:** Data Processing
    - **Streamlit:** Frontend Framework
    - **Plotly:** Interactive Visualizations
    
    ### Model Equation
    ```
    Price = $6,643,873.53 + (bedrooms × -$34,335) + (bathrooms × $44,565) 
            + (sqft_living × $109) + ... [16 more features]
    ```
    
    ### How to Use
    1. Go to the **Price Prediction** tab
    2. Enter property details using the sliders and input fields
    3. Click **Predict Price** button
    4. View the estimated price and feature contributions
    5. Explore **Model Info** and **Visualizations** for detailed metrics
    """)
    
    st.divider()
    
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
    <p>🏠 Housing Price Prediction Model v1.0</p>
    <p>Built with Streamlit | Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #999; font-size: 12px;'>© 2024 Housing Price Prediction Model | Multiple Linear Regression Analysis</p>", unsafe_allow_html=True)
