import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import joblib


# Load the KNN model
LRM = joblib.load('LRM_model.joblib')

def main():
    # Set page title and favicon
    st.set_page_config(page_title='Risk of Coronary Heart Disease', page_icon=':heart:')
    
    # Set app title
    st.title('Risk of Coronary Heart Disease')
    
    # Create form to take user input
    with st.form(key='heart_disease_prediction_form'):
        # Add form fields for user input
        st.write('Please fill in the following details to check whether you are at risk of heart disease or not:')
        age = st.selectbox('Select your age:', options=range(1, 101))
        gender = st.selectbox(label="Select your gender:", options=['Male', 'Female'])
        cigsPerDay = st.selectbox(label="Enter number of cigarettes smoked per day:", options=range(1, 101))
        sysBP = st.selectbox(label="Enter systolic blood pressure:", options=range(1, 300))
        diaBP = st.selectbox(label="Enter diastolic blood pressure:", options=range(1, 200))
        totChol = st.selectbox(label="Enter total cholesterol level:", options=range(50, 600))
        prevalentHyp = st.selectbox(label="Do you have hypertension?", options=['Yes', 'No'])
        diabetes = st.selectbox(label="Do you have diabetes?", options=['Yes', 'No'])
        glucose = st.selectbox(label="Enter fasting blood sugar level:", options=range(50, 500))
        BPMeds = st.selectbox(label="Are you on blood pressure medication?", options=['Yes', 'No'])
        
        # Add submit button to form
        submitted = st.form_submit_button(label='Submit')
        
        # If user submits form
        if submitted:
            # Create a dictionary from the user input and convert it into a DataFrame
            my_data = {'age': [age],
                       'male': [1] if gender == 'Male' else [0],
                       'cigsPerDay': [cigsPerDay],
                       'sysBP': [sysBP],
                       'diaBP': [diaBP],
                       'totChol': [totChol],
                       'prevalentHyp': [1] if prevalentHyp == 'Yes' else [0],
                       'diabetes': [1] if diabetes == 'Yes' else [0],
                       'glucose': [glucose],
                       'BPMeds': [1] if BPMeds == 'Yes' else [0]}
            
            my_df = pd.DataFrame(my_data)
            
            # Scale the input data using MinMax Scaler
            scaler = MinMaxScaler() 
            my_df_scaled = pd.DataFrame(scaler.fit_transform(my_df), columns=my_df.columns)
            
            # Make prediction using LRM model
            my_y_pred = LRM.predict(my_df)
            
            # Display prediction result
            if my_y_pred[0] == 1:
                st.write("You are at risk of developing heart disease.")
            else:
                st.write("You are not at risk of developing heart disease.")

if __name__ == '__main__':
    main()

