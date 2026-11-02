import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Used Car Price Predictor")


@st.cache_resource
def load_pipeline():
    return joblib.load("used_car_model.joblib")


@st.cache_data
def load_options():
    with open("app_options.json") as f:
        return json.load(f)


pipe = load_pipeline()
opt = load_options()

st.title("Used Car Price Predictor")
st.write("Enter the car's details to get an estimated asking price in rupees.")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", opt["Brand"])
    car_model = st.selectbox("Model", opt["model"])
    transmission = st.selectbox("Transmission", opt["Transmission"])
    fuel = st.selectbox("Fuel type", opt["FuelType"])

with col2:
    owner = st.selectbox("Owner", opt["Owner"])
    age = st.number_input("Age (years)", min_value=opt["Age_min"],
                          max_value=opt["Age_max"], value=5, step=1)
    km = st.number_input("Kilometres driven", min_value=0,
                         max_value=opt["km_max"], value=60000, step=1000)

if st.button("Predict price"):
    row = pd.DataFrame([{
        "Brand": brand,
        "model": car_model,
        "Age": float(age),
        "kmDriven": float(km),
        "Transmission": transmission,
        "FuelType": fuel,
        "Owner": owner,
    }])
    price = np.exp(pipe.predict(row)[0])
    st.success(f"Estimated price: ₹ {price:,.0f}")
    st.caption("The typical error of this model on unseen cars is about 17% of the price.")