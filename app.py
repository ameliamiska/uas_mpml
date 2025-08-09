# =========================
# app.py (VERSI PERBAIKI)
# =========================

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Profit Menu", page_icon="🍽️")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    try:
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Verifikasi model
        if not hasattr(model, 'predict'):
            raise ValueError("Model tidak valid")
            
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

model = load_model()

# =========================
# MAPPING UNTUK ENCODING MANUAL
# =========================
resto_mapping = {'R001': 0, 'R002': 1, 'R003': 2}
category_mapping = {'Appetizer': 0, 'Beverages': 1, 'Dessert': 2, 'Main Course': 3}
item_mapping = {
    'Bruschetta': 0, 'Caprese Salad': 1, 'Spinach Artichoke Dip': 2, 'Stuffed Mushrooms': 3,
    'Coffee': 4, 'Iced Tea': 5, 'Lemonade': 6, 'Soda': 7,
    'Chocolate Lava Cake': 8, 'Fruit Tart': 9, 'New York Cheesecake': 10, 'Tiramisu': 11,
    'Chicken Alfredo': 12, 'Grilled Steak': 13, 'Shrimp Scampi': 14, 'Vegetable Stir-Fry': 15
}

# =========================
# JUDUL
# =========================
st.title("🍽️ Restaurant Menu Optimization")
st.write("Aplikasi ini memprediksi profitabilitas menu berdasarkan kategori dan harga.")

# =========================
# INPUT USER
# =========================
st.sidebar.header("Input Data")

def user_input_features():
    restaurant_id = st.sidebar.selectbox('Restaurant ID', ['R001', 'R002', 'R003'])
    menu_category = st.sidebar.selectbox('Menu Category', ['Appetizer', 'Beverages', 'Dessert', 'Main Course'])
    menu_item = st.sidebar.selectbox('Menu Item', list(item_mapping.keys()))
    price = st.sidebar.number_input('Price ($)', min_value=0.0, value=10.0, step=0.5, format='%f')
    
    return {
        'RestaurantID': restaurant_id,
        'MenuCategory': menu_category,
        'MenuItem': menu_item,
        'Price': price
    }

input_data = user_input_features()

# =========================
# PREPROCESSING & PREDIKSI
# =========================
def preprocess_input(data):
    """Konversi input ke format yang diterima model"""
    return np.array([
        resto_mapping[data['RestaurantID']],
        category_mapping[data['MenuCategory']],
        item_mapping[data['MenuItem']],
        data['Price']
    ]).reshape(1, -1)

if st.sidebar.button("Predict Profitability") and model is not None:
    try:
        # Preprocessing input
        features = preprocess_input(input_data)
        
        # Prediksi
        prediction = model.predict(features)
        result_map = {0: "🟢 Low Profit", 1: "🟡 Medium Profit", 2: "🔴 High Profit"}
        
        st.subheader("Hasil Prediksi")
        st.success(f"📊 Prediksi: **{result_map[int(prediction[0])]}**")
        
        # Tampilkan probabilitas jika tersedia
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            st.write("Probabilitas:")
            st.write(f"- Low: {proba[0]:.1%}")
            st.write(f"- Medium: {proba[1]:.1%}")
            st.write(f"- High: {proba[2]:.1%}")
            
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {str(e)}")
        st.info("""
        Kemungkinan penyebab:
        1. Format input tidak sesuai
        2. Model tidak dikenali
        3. Error preprocessing
        """)

# =========================
# TAMPILKAN DATA INPUT
# =========================
st.subheader("Data Input")
st.json(input_data)

# Debug info
if st.checkbox("Tampilkan Info Debug"):
    st.subheader("Debug Information")
    if model is not None:
        st.write("Tipe Model:", type(model))
        st.write("Parameter Model:", model.get_params())
    else:
        st.warning("Model tidak tersedia")