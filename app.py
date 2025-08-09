# =========================
# app.py
# =========================
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(page_title="Prediksi Profit Menu", page_icon="🍽️")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)

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
    return pd.DataFrame([{
        'RestaurantID': restaurant_id,
        'MenuCategory': menu_category,
        'MenuItem': menu_item,
        'Price': price
    }])

input_df = user_input_features()

# =========================
# PREDIKSI
# =========================
if st.sidebar.button("Predict Profitability"):
    # Mapping kategori sesuai training
    mapping_restaurant = {'R001': 0, 'R002': 1, 'R003': 2}
    mapping_category = {'Appetizer': 0, 'Beverages': 1, 'Dessert': 2, 'Main Course': 3}
    mapping_item = {
        'Bruschetta': 0,'Caprese Salad': 1,'Spinach Artichoke Dip': 2,'Stuffed Mushrooms': 3,
        'Coffee': 4,'Iced Tea': 5,'Lemonade': 6,'Soda': 7,
        'Chocolate Lava Cake': 8,'Fruit Tart': 9,'New York Cheesecake': 10,'Tiramisu': 11,
        'Chicken Alfredo': 12,'Grilled Steak': 13,'Shrimp Scampi': 14,'Vegetable Stir-Fry': 15
    }

    # Encode input user
    input_df['RestaurantID'] = input_df['RestaurantID'].map(mapping_restaurant)
    input_df['MenuCategory'] = input_df['MenuCategory'].map(mapping_category)
    input_df['MenuItem'] = input_df['MenuItem'].map(mapping_item)

    # Scaling price (pakai scaling dari training)
    # NOTE: Karena kita nggak simpan scaler, kita fit ulang pakai nilai mean & std saat training
    # Kalau mau lebih presisi, simpan scaler saat training dan load di sini
    price_mean = 0.0   # ganti sesuai mean harga dari training
    price_std = 1.0    # ganti sesuai std harga dari training
    input_df['Price'] = (input_df['Price'] - price_mean) / price_std

    # Prediksi
    prediction = model.predict(input_df)
    result_map = {0: "Low Profit", 1: "Medium Profit", 2: "High Profit"}

    st.subheader("Hasil Prediksi")
    st.success(f"📊 Prediksi: **{result_map[int(prediction[0])]}**")

# =========================
# TAMPILKAN DATA INPUT
# =========================
st.subheader("Data Input")
st.write(input_df)
