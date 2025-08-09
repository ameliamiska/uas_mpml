# =========================
# app.py (VERSI PERBAIKAN)
# =========================

import streamlit as st
import pickle
import pandas as pd

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Profit Menu", page_icon="🍽️")

# =========================
# LOAD MODEL & FITUR
# =========================
@st.cache_resource
def load_model_and_features():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    # Dapatkan nama fitur dari preprocessor
    try:
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    except AttributeError:
        feature_names = ['RestaurantID', 'MenuCategory', 'MenuItem', 'Price']
    return model, feature_names

model, expected_features = load_model_and_features()

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
    try:
        # Pastikan urutan kolom sesuai training
        input_df = input_df[['RestaurantID', 'MenuCategory', 'MenuItem', 'Price']]
        
        # Lakukan prediksi
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[0]
        
        # Mapping hasil
        result_map = {
            0: "🟢 Low Profit", 
            1: "🟡 Medium Profit", 
            2: "🔴 High Profit"
        }
        
        # Tampilkan hasil
        st.subheader("Hasil Prediksi")
        st.success(f"📊 **{result_map[int(prediction[0])]}**")
        
        # Tampilkan probabilitas
        st.progress(proba.max())
        st.write(f"""
        - Probabilitas Low: {proba[0]:.1%}
        - Probabilitas Medium: {proba[1]:.1%}
        - Probabilitas High: {proba[2]:.1%}
        """)
        
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {str(e)}")
        st.info("Pastikan semua input sudah diisi dengan benar")

# =========================
# TAMPILKAN DATA INPUT
# =========================
st.subheader("Data Input")
st.write(input_df)

# Debugging info
if st.checkbox("Show debug info"):
    st.subheader("Debug Information")
    st.write("Expected features by model:", expected_features)
    st.write("Input data structure:", input_df.dtypes)
    st.json(input_df.iloc[0].to_dict())