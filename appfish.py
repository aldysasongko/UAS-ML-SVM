import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Memuat model dan scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')  # Pastikan label encoder disimpan jika belum

# Judul Aplikasi
st.title("Prediksi Spesies Ikan menggunakan SVM")

# Input dari pengguna
st.header("Masukkan Data Ikan")
length = st.number_input("Panjang Ikan (cm)", value=0.0, format="%.2f")
weight = st.number_input("Berat Ikan (kg)", value=0.0, format="%.2f")
w_l_ratio = st.number_input("Rasio Berat/Panjang", value=0.0, format="%.2f")

if st.button('Prediksi'):
    # Menyiapkan data input
    input_data = np.array([[length, weight, w_l_ratio]])

    # Lakukan scaling pada data input
    input_scaled = scaler.transform(input_data)

    # Prediksi spesies
    prediction = model.predict(input_scaled)
    predicted_species = label_encoder.inverse_transform(prediction)

    # Tampilkan hasil prediksi
    st.success(f"Spesies ikan yang diprediksi: {predicted_species[0]}")
