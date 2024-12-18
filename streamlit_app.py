import streamlit as st
import pandas as pd
import pickle
import os

# Function to load models with error handling
def load_model(filename):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"Error: {filename} not found. Ensure the model file is uploaded.")
        return None

# Load the models
lgbm_model = load_model('lightgbm_model.pkl')
rf_model = load_model('random_forest_model.pkl')
xgb_model = load_model('xgboost_model.pkl')

# Function to get predictions from all models
def get_predictions(input_data):
    lgbm_pred = lgbm_model.predict(input_data)[0] if lgbm_model else None
    rf_pred = rf_model.predict(input_data)[0] if rf_model else None
    xgb_pred = xgb_model.predict(input_data)[0] if xgb_model else None
    return lgbm_pred, rf_pred, xgb_pred

# Function to display the front page
def front_page():
    st.title("New York Workers' Compensation Board (WCB) Project")

    if os.path.exists("WorkersClaimForm.jpg"):
        st.image("WorkersClaimForm.jpg", use_column_width=True)
    else:
        st.warning("WorkersClaimForm.jpg not found. Please upload the image to the repository.")

    st.write("""
    ## Project Introduction
    "The New York Workers’ Compensation Board (WCB) administers and regulates workers’ compensation, disability, volunteer firefighters, volunteer ambulance workers, and volunteer civil defence workers’ benefits. As the regulating authority, the WCB is responsible for assembling and deciding on claims whenever it becomes aware of a workplace injury. Since 2000, the WCB has assembled and reviewed more than 5 million claims. However, manually reviewing all claims is an arduous and time-consuming process. For that reason, the WCB has reached out to Nova IMS to assist them in the creation of a model that can automate the decision-making whenever a new claim is received."
    """)

    st.write("""
    ## Group Members
    - **André Oliveira** (20211539)  
    - **Bernardo Faria** (20240579)  
    - **Hassan Bhatti** (20241023)  
    - **João Marto** (20211618)  
    - **Miguel Mangerona** (20240595)
    """)

# Function to display the prediction page
def prediction_page():
    st.title("WCB Claim Type Prediction")

    # Sidebar for example selection
    st.sidebar.header("Load Example Data")

    example_data = {
        "Example 1": {
            'Attorney/Representative': 1,
            'Average Weekly Wage': 1200,
            'Birth Year': 1985,
            'Days To Process Report': 20,
            'IME-4 Count': 3,
            'Medical Fee Region': 2,
            'WCIO Nature of Injury Code': 5,
            'WCIO Part Of Body Code': 6,
            'Claim Report Received': 1,
            'Hearing Held': 0,
            'Carrier': 2
        }
    }

    selected_example = st.sidebar.selectbox("Select an Example", list(example_data.keys()))

    if st.sidebar.button("Load Example"):
        st.session_state['input_data'] = example_data[selected_example]

    # Initialize input data
    if 'input_data' not in st.session_state:
        st.session_state['input_data'] = {}

    # User input fields
    st.write("### Enter claim details below to predict the claim type.")

    input_data = {}
    input_data['Attorney/Representative'] = st.selectbox('Attorney/Representative', [0, 1], index=st.session_state['input_data'].get('Attorney/Representative', 0))
    input_data['Average Weekly Wage'] = st.number_input('Average Weekly Wage', min_value=0, value=st.session_state['input_data'].get('Average Weekly Wage', 0))

    # Predict button
    if st.button("Predict"):
        df = pd.DataFrame([input_data])
        lgbm_pred, rf_pred, xgb_pred = get_predictions(df)

        st.write("### Predictions")
        st.write(f"LightGBM Prediction: {lgbm_pred}")
        st.write(f"Random Forest Prediction: {rf_pred}")
        st.write(f"XGBoost Prediction: {xgb_pred}")

# Navigation
menu = st.sidebar.selectbox("Choose a Page", ["Home", "Prediction"])

if menu == "Home":
    front_page()
else:
    prediction_page()
