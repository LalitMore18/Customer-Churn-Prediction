import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

model = tf.keras.models.load_model('churn_model.h5')

with open('label_encoder_gen.pkl', 'rb') as f:
    label_encoder_gen = pickle.load(f)

with open('onehot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gen.classes_)
age = st.slider('Age', 18, 92, 30)
balance = st.number_input('Balance', value=10000.0)
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary', value=50000.0)
tenure = st.slider('Tenure', 0, 10, 1)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender' : [label_encoder_gen.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = one_hot_encoder.transform([[geography]])
geo_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder.get_feature_names_out())
input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
churn_prob = prediction[0][0]

if churn_prob > 0.5:
    st.write(f'Customer is likely to churn with a probability of {churn_prob:.2f}')        
else:
    st.write(f'Customer is unlikely to churn with a probability of {churn_prob:.2f}')




