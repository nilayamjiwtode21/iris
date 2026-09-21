import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("iris_model.pki")

st.title("Iris Flower Prediction")

st.write("Enter the flower measurements below:")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0)

# Prediction button
if st.button("Predict"):
    input_data = np.array([
        [sepal_length, sepal_width, petal_length, petal_width]
    ])

    prediction = model.predict(input_data)

    st.success(f"Predicted Iris species: {prediction[0]}")
