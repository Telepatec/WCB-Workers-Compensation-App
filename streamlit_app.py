import streamlit as st

st.title('🤖 Machine Learning App')

st.info('This app builds a machine learning model!')

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the models
with open('lightgbm_model.pkl', 'rb') as file:
    lgbm_model = pickle.load(file)

with open('random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('xgboost_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

# Function to get predictions from all models
def get_predictions(input_data):
    lgbm_pred = lgbm_model.predict(input_data)[0]
    rf_pred = rf_model.predict(input_data)[0]
    xgb_pred = xgb_model.predict(input_data)[0]
    return lgbm_pred, rf_pred, xgb_pred

# Function to display the front page
def front_page():
    st.title("New York Workers' Compensation Board (WCB) Project")

    st.image("WorkersClaimForm.jpg", use_column_width=True)

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
        },
        "Example 2": {
            'Attorney/Representative': 0,
            'Average Weekly Wage': 0,
            'Birth Year': 1970,
            'Days To Process Report': 300,
            'IME-4 Count': 0,
            'Medical Fee Region': 4,
            'WCIO Nature of Injury Code': 2,
            'WCIO Part Of Body Code': 4,
            'Claim Report Received': 0,
            'Hearing Held': 0,
            'Carrier': 0
        },
        "Example 3": {
            'Attorney/Representative': 1,
            'Average Weekly Wage': 3500,
            'Birth Year': 1990,
            'Days To Process Report': 45,
            'IME-4 Count': 5,
            'Medical Fee Region': 1,
            'WCIO Nature of Injury Code': 6,
            'WCIO Part Of Body Code': 3,
            'Claim Report Received': 1,
            'Hearing Held': 1,
            'Carrier': 4
        },
        "Example 4": {
            'Attorney/Representative': 1,
            'Average Weekly Wage': 600,
            'Birth Year': 2000,
            'Days To Process Report': 10,
            'IME-4 Count': 1,
            'Medical Fee Region': 3,
            'WCIO Nature of Injury Code': 3,
            'WCIO Part Of Body Code': 7,
            'Claim Report Received': 1,
            'Hearing Held': 0,
            'Carrier': 1
        },
        "Example 5": {
            'Attorney/Representative': 1,
            'Average Weekly Wage': 8000,
            'Birth Year': 1960,
            'Days To Process Report': 90,
            'IME-4 Count': 8,
            'Medical Fee Region': 4,
            'WCIO Nature of Injury Code': 1,
            'WCIO Part Of Body Code': 1,
            'Claim Report Received': 1,
            'Hearing Held': 1,
            'Carrier': 3
        }
    }

    selected_example = st.sidebar.selectbox("Select an Example", list(example_data.keys()))

    # Load the example data into session state
    if st.sidebar.button("Load Example"):
        st.session_state['input_data'] = example_data[selected_example]

    # Initialize input data
    if 'input_data' not in st.session_state:
        st.session_state['input_data'] = {}

    # User input fields
    st.write("### Enter claim details below to predict the claim type.")

    input_data = {}
    for field, value in example_data[selected_example].items():
        input_data[field] = st.number_input(f"{field}", value=value)

    # Predict button
    if st.button("Predict"):
        df = pd.DataFrame([input_data])
        lgbm_pred, rf_pred, xgb_pred = get_predictions(df)

        # Map predictions to their labels
        label_mapping = {
            1: '1. CANCELLED',
            2: '2. NON-COMP',
            3: '3. MED ONLY',
            4: '4. TEMPORARY',
            5: '5. PPD SCH LOSS',
            6: '6. PPD NSL',
            7: '7. PTD',
            8: '8. DEATH'
        }

        predictions = {
            'LightGBM': label_mapping[lgbm_pred],
            'Random Forest': label_mapping[rf_pred],
            'XGBoost': label_mapping[xgb_pred]
        }

        # Display the predictions
        st.subheader("Predictions")
        results_df = pd.DataFrame(list(predictions.items()), columns=["Model", "Prediction"])
        st.table(results_df)

# Function to display the visualization page
def visualization_page():
    st.title("LightGBM's Confusion Matrix")

    st.markdown("""
    ### Important Observations to Take Away:
                
    - The model struggles with some classes, often predicting "2. NON-COMP" instead of "1. CANCELLED".
    - "2. NON-COMP" and "4. TEMPORARY" have high accuracy.
    - Class "7. PTD" is rarely predicted correctly due to fewer observations.
    """)

    st.image("output.png", use_column_width=True)

# Navigation
PAGES = {
    "Front Page": front_page,
    "Prediction Page": prediction_page,
    "Visualizations": visualization_page
}

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[choice]()
