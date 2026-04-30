import streamlit as st
import pandas as pd
import pickle   
import numpy as np

model = pickle.load(open(r"C:\Users\raavi\Agentic AI\Ml Projects\Mutiple Linear Regression\CompanyInvestment\linear_regression_model_companyinvestment_pred.pkl", 'rb'))

st.title("Company Investment Prediction")
st.write("Enter the details of the company to predict the investment amount.")  

Enter_the_amount_for_Digital_marketing = st.number_input("Digital Marketing Amount", min_value=0, max_value=1000000, step=1000)
Enter_the_amount_for_Promotion = st.number_input("Promotion", min_value=0, max_value=1000000, step=1000)
Enter_the_amount_for_Research = st.number_input("Research", min_value=0, max_value=1000000, step=1000)
state_options = ["Hyderabad", "Bangalore", "Chennai"]
selected_state = st.selectbox("State", state_options)

if st.button("Predict Investment"):
    state_bangalore = 1 if selected_state == "Bangalore" else 0
    state_chennai = 1 if selected_state == "Chennai" else 0
    state_hyderabad = 1 if selected_state == "Hyderabad" else 0

    input_data = np.array([[
        Enter_the_amount_for_Digital_marketing,
        Enter_the_amount_for_Promotion,
        Enter_the_amount_for_Research,
        state_bangalore,
        state_chennai,
        state_hyderabad
    ]])

    predicted_investment = model.predict(input_data)

    st.success(f"Predicted Investment Amount: {predicted_investment[0]:.2f}")

st.write("This application uses a linear regression model to predict the investment amount based on the input features. Please enter valid numerical values for the digital marketing, promotion, and research amounts, and select a state to get the prediction.")

