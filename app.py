# =========================
# app.py
# =========================

import streamlit as st
import joblib
import pandas as pd

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Profit Menu", page_icon="🍽️")

# =========================
# LOAD PIPELINE MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")  

model = load_model()

# =========================
# JUDUL
# =========================
st.title("🍽️ Restaurant Menu Optimization")
st.write("Aplikasi ini memprediksi profitabilitas menu berdasarkan kategori dan harga.")

# =========================
# INPUT USER
# =========================
st.sidebar.header("Input Data")

restaurant_id = st.sidebar.selectbox('Restaurant ID', ['R001','R002','R003'])
menu_category = st.sidebar.selectbox('Menu Category', ['Appetizer', 'Beverages', 'Dessert', 'Main Course'])
menu_item = st.sidebar.selectbox('Menu Item', [
    'Bruschetta','Caprese Salad','Spinach Artichoke Dip','Stuffed Mushrooms',
    'Coffee','Iced Tea','Lemonade','Soda',
    'Chocolate Lava Cake','Fruit Tart','New York Cheesecake','Tiramisu',
    'Chicken Alfredo','Grilled Steak','Shrimp Scampi','Vegetable Stir-Fry'
])
price = st.sidebar.number_input('Price ($)', min_value=0.0, value=10.0, step=0.5)

# Buat DataFrame sesuai format training
input_df = pd.DataFrame([{
    'RestaurantID': restaurant_id,
    'MenuCategory': menu_category,
    'MenuItem': menu_item,
    'Price': price
}])

# =========================
# PREDIKSI
# =========================
if st.sidebar.button("Predict Profitability"):
    prediction = model.predict(input_df)[0]
    result_map = {0: "Low Profit", 1: "Medium Profit", 2: "High Profit"}

    st.subheader("Hasil Prediksi")
    st.success(f"📊 Prediksi: **{result_map[prediction]}**")

# =========================
# TAMPILKAN DATA INPUT
# =========================
st.subheader("Data Input")
st.write(input_df)
