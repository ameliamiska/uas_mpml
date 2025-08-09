import streamlit as st
import pandas as pd
import pickle

# =========================
# LOAD PIPELINE MODEL
# =========================
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# =========================
# TITLE
# =========================
st.title("🍽 Prediksi Profitabilitas Menu Restoran")
st.write("Aplikasi ini memprediksi apakah menu restoran menguntungkan atau tidak berdasarkan data input.")

# =========================
# INPUT USER
# =========================
restaurant_id = st.selectbox("Pilih Restaurant ID", ["R001", "R002", "R003"])
menu_category = st.selectbox("Pilih Menu Category", ["Appetizer", "Main Course", "Dessert", "Beverage"])
menu_item = st.text_input("Nama Menu Item", "Bruschetta")
price = st.number_input("Harga ($)", min_value=1.0, max_value=100.0, step=0.5)

# =========================
# PREDIKSI
# =========================
if st.button("Predict Profitability"):
    # Buat DataFrame sesuai format pipeline
    input_df = pd.DataFrame({
        "Restaurant ID": [restaurant_id],
        "Menu Category": [menu_category],
        "Menu Item": [menu_item],
        "Price": [price]
    })

    prediction = model.predict(input_df)[0]
    
    if prediction == 1:
        st.success("✅ Menu ini DIPREDIKSI MENGUNTUNGKAN")
    else:
        st.error("⚠️ Menu ini DIPREDIKSI TIDAK MENGUNTUNGKAN")

# =========================
# INFO MODEL
# =========================
st.sidebar.header("Info Model")
st.sidebar.write("Model ini menggunakan Pipeline: Preprocessing (OneHotEncoder) + RandomForestClassifier.")
st.sidebar.write("File model: best_model.pkl")
