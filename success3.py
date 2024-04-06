import streamlit as st
import pandas as pd
import pickle
import shap
import shap.plots
import matplotlib.pyplot as plt

# Title
st.header("Streamlit Machine Learning App")

# Input bar 1
Stone_volume = st.number_input("Enter Stone volume")

# Input bar 2
Mean_IRP = st.number_input("Enter Mean intrarenal pressures (IRP)")

# Input bar 3
Operation_time = st.number_input("Enter Operation time")

# Dropdown input4
Urine_WBC = st.selectbox("Select Urine WBC Inspection Results", ("positive", "negative"))

# Dropdown input5
Urine_nitrite = st.selectbox("Select Urine Nitrite Inspection Results", ("Positive", "Negative"))

# Input bar 6
Radiomics_score = st.number_input("Enter Radiomics scores")

# If button is pressed
if st.button("Predict"):
    # Unpickle classifier
    pickle_in1 = open('classifier4.pkl', 'rb')
    classifier4 = pickle.load(pickle_in1)

    # Store inputs into dataframe
    X = pd.DataFrame([[Stone_volume,Mean_IRP, Operation_time, Urine_WBC, Urine_nitrite, Radiomics_score]],
                     columns=["Stone_volume", "Mean_IRP", "Operation_time", "Urine_WBC", "Urine_nitrite", "Radiomics_score"])
    X = X.replace(["positive", "negative"], [1, 0])
    X = X.replace(["Positive", "Negative"], [1, 0])

    # Get prediction probability
    prediction_proba = classifier4.predict_proba(X)

    # Output prediction probability
    st.text(f"The probability of developing into urosepsis is {prediction_proba[0, 1]}")

    # SHAP explanation
    explainer = shap.TreeExplainer(classifier4)
    shap_values = explainer.shap_values(X)

    # Plot SHAP decision plot
    expected_value = explainer.expected_value
    fig, ax = plt.subplots()
    shap.decision_plot(expected_value, shap_values, X)
    st.pyplot(fig)