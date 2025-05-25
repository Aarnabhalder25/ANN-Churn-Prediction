import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Page config
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🔮", layout="centered")

# Load model and preprocessing tools
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# --- Header ---
st.title("🔮 Customer Churn Prediction")
st.markdown("Use this app to predict the likelihood of a customer churning based on input attributes.")

st.markdown("---")

# --- Input Section ---
st.subheader("📋 Enter Customer Details")

col1, col2 = st.columns(2)
with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, help="Customer's age")

with col2:
    credit_score = st.number_input('📊 Credit Score', min_value=300.0, max_value=1000.0)
    estimated_salary = st.number_input('💰 Estimated Salary')
    balance = st.number_input('🏦 Account Balance')

col3, col4, col5 = st.columns(3)
with col3:
    tenure = st.slider('⏳ Tenure (Years)', 0, 10)
with col4:
    num_of_products = st.slider('📦 Number of Products', 1, 4)
with col5:
    has_cr_card = st.selectbox('💳 Has Credit Card?', ['No', 'Yes'])
    is_active_member = st.selectbox('✅ Active Member?', ['No', 'Yes'])

# Encode categorical and numerical inputs
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
    'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

input_scaled = scaler.transform(input_data)

# --- Prediction Section ---
if st.button("🚀 Predict"):
    prediction = model.predict(input_scaled)
    probability = prediction[0][0]

    st.markdown("---")
    st.subheader("📈 Prediction Result")

    st.progress(float(min(probability, 1.0)))
    st.write(f"**Churn Probability:** `{probability:.2f}`")

    if probability > 0.5:
        st.error("⚠️ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **not likely to churn**.")

# --- Footer ---
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
