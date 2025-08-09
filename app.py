# =========================
# app.py (SIMPLIFIED VERSION)
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
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

model = load_model()

# =========================
# MAPPING UNTUK ENCODING
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
    menu_item = st.sidebar.selectbox('Menu Item', [
        'Bruschetta','Caprese Salad','Spinach Artichoke Dip','Stuffed Mushrooms',
        'Coffee','Iced Tea','Lemonade','Soda',
        'Chocolate Lava Cake','Fruit Tart','New York Cheesecake','Tiramisu',
        'Chicken Alfredo','Grilled Steak','Shrimp Scampi','Vegetable Stir-Fry'
    ])
    price = st.sidebar.number_input('Price ($)', min_value=0.0, value=10.0, step=0.5, format='%f')
    return {
        'RestaurantID': restaurant_id,
        'MenuCategory': menu_category,
        'MenuItem': menu_item,
        'Price': price
    }

# =========================
# PREPROCESSING
# =========================
def preprocess_data(input_data):
    """Convert input data to model format"""
    return np.array([
        resto_mapping[input_data['RestaurantID']],
        category_mapping[input_data['MenuCategory']],
        item_mapping[input_data['MenuItem']],
        input_data['Price']
    ]).reshape(1, -1)

# =========================
# TAMPILAN UTAMA
# =========================
input_data = user_input_features()
input_df = pd.DataFrame([input_data])

if st.sidebar.button("Predict Profitability"):
    if model is not None:
        try:
            # Preprocess and predict
            features = preprocess_data(input_data)
            prediction = model.predict(features)
            
            # Display results
            result_map = {0: "Low Profit", 1: "Medium Profit", 2: "High Profit"}
            st.subheader("Hasil Prediksi")
            st.success(f"📊 Prediksi: **{result_map[int(prediction[0])]}**")
            
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {str(e)}")
    else:
        st.error("Model tidak tersedia")

# Tampilkan data input
st.subheader("Data Input")
st.write(input_df)