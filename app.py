# =========================
# app.py (VERSI AMAN)
# =========================

import streamlit as st
import pickle
import pandas as pd

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Profit Menu", page_icon="🍽️")

# =========================
# LOAD MODEL (DENGAN ERROR HANDLING)
# =========================
@st.cache_resource
def load_model():
    try:
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)
        st.success("✅ Model berhasil dimuat")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

model = load_model()

# Fitur yang diharapkan (sesuaikan dengan training)
EXPECTED_FEATURES = ['RestaurantID', 'MenuCategory', 'MenuItem', 'Price']

# =========================
# JUDUL
# =========================
st.title("🍽️ Restaurant Menu Optimization")

# =========================
# INPUT USER
# =========================
def user_input_features():
    st.sidebar.header("Input Data")
    data = {
        'RestaurantID': st.sidebar.selectbox('Restaurant ID', ['R001', 'R002', 'R003']),
        'MenuCategory': st.sidebar.selectbox('Menu Category', ['Appetizer', 'Beverages', 'Dessert', 'Main Course']),
        'MenuItem': st.sidebar.selectbox('Menu Item', [
            'Bruschetta','Caprese Salad','Spinach Artichoke Dip','Stuffed Mushrooms',
            'Coffee','Iced Tea','Lemonade','Soda',
            'Chocolate Lava Cake','Fruit Tart','New York Cheesecake','Tiramisu',
            'Chicken Alfredo','Grilled Steak','Shrimp Scampi','Vegetable Stir-Fry'
        ]),
        'Price': st.sidebar.number_input('Price ($)', min_value=0.0, value=10.0, step=0.5)
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# =========================
# PREDIKSI
# =========================
if model is not None and st.sidebar.button("Predict Profitability"):
    try:
        # Pastikan kolom sesuai
        input_df = input_df[EXPECTED_FEATURES]
        
        # Prediksi
        prediction = model.predict(input_df)
        result_map = {0: "Low Profit", 1: "Medium Profit", 2: "High Profit"}
        
        st.subheader("Hasil Prediksi")
        st.success(f"📊 Prediksi: **{result_map.get(int(prediction[0]), 'Unknown')}**")
        
        # Jika model support probabilitas
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)[0]
            st.write(f"Probabilitas: {dict(zip(['Low','Medium','High'], [f'{p:.1%}' for p in proba]))}")
    
    except Exception as e:
        st.error(f"Error saat prediksi: {str(e)}")
        st.info("Pastikan model dan input sesuai")

# Tampilkan input
st.subheader("Data Input")
st.write(input_df)