# =========================
# app.py (FIXED VERSION)
# =========================
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Profit Menu", page_icon="🍽️")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    try:
        with open("new_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        if not hasattr(model, 'predict'):
            raise ValueError("Invalid model - missing predict method")
            
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_model()

# =========================
# PREPROCESSING CONFIG
# =========================
# Mapping values (should match training)
RESTAURANT_MAP = {'R001': 1, 'R002': 2, 'R003': 3}
CATEGORY_MAP = {
    'Appetizer': 1, 
    'Beverages': 2, 
    'Dessert': 3, 
    'Main Course': 4
}
ITEM_MAP = {
    'Bruschetta': 1, 'Caprese Salad': 2, 'Spinach Artichoke Dip': 3, 'Stuffed Mushrooms': 4,
    'Coffee': 5, 'Iced Tea': 6, 'Lemonade': 7, 'Soda': 8,
    'Chocolate Lava Cake': 9, 'Fruit Tart': 10, 'New York Cheesecake': 11, 'Tiramisu': 12,
    'Chicken Alfredo': 13, 'Grilled Steak': 14, 'Shrimp Scampi': 15, 'Vegetable Stir-Fry': 16
}

# Price normalization (use values from your training data)
PRICE_MEAN = 12.5  # Replace with actual mean from training
PRICE_STD = 4.2     # Replace with actual std from training

# =========================
# PREPROCESSING FUNCTION
# =========================
def preprocess_input(input_data):
    """Convert raw input to model-ready format"""
    try:
        # Convert categorical features
        restaurant_encoded = RESTAURANT_MAP.get(input_data['RestaurantID'], 0)
        category_encoded = CATEGORY_MAP.get(input_data['MenuCategory'], 0)
        item_encoded = ITEM_MAP.get(input_data['MenuItem'], 0)
        
        # Normalize price
        price_normalized = (input_data['Price'] - PRICE_MEAN) / PRICE_STD
        
        # Ensure correct feature order
        return np.array([
            restaurant_encoded,
            category_encoded,
            item_encoded,
            price_normalized
        ]).reshape(1, -1)
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None

# =========================
# UI COMPONENTS
# =========================
st.title("🍽️ Restaurant Menu Optimization")
st.write("Aplikasi ini memprediksi profitabilitas menu berdasarkan kategori dan harga.")

def user_input_features():
    st.sidebar.header("Input Data")
    return {
        'RestaurantID': st.sidebar.selectbox('Restaurant ID', list(RESTAURANT_MAP.keys())),
        'MenuCategory': st.sidebar.selectbox('Menu Category', list(CATEGORY_MAP.keys())),
        'MenuItem': st.sidebar.selectbox('Menu Item', list(ITEM_MAP.keys())),
        'Price': st.sidebar.number_input('Price ($)', min_value=0.0, value=10.0, step=0.5)
    }

# =========================
# PREDICTION LOGIC
# =========================
input_data = user_input_features()

if st.sidebar.button("Predict Profitability") and model:
    try:
        # Preprocess input
        features = preprocess_input(input_data)
        
        if features is not None:
            # Get prediction
            prediction = model.predict(features)
            proba = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None
            
            # Display results
            st.subheader("Hasil Prediksi")
            
            result_map = {
                0: ("🔴 Low Profit", "red"),
                1: ("🟡 Medium Profit", "orange"),
                2: ("🟢 High Profit", "green")
            }
            
            pred_text, pred_color = result_map.get(int(prediction[0]), ("🔵 Unknown", "blue")
            st.success(f"📊 **{pred_text}**")
            
            if proba is not None:
                st.write("Probabilitas:")
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Low", f"{proba[0]:.1%}")
                with cols[1]:
                    st.metric("Medium", f"{proba[1]:.1%}")
                with cols[2]:
                    st.metric("High", f"{proba[2]:.1%}")
                
                # Visualize probabilities
                st.progress(max(proba))
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# =========================
# DISPLAY INPUTS
# =========================
st.subheader("Data Input")
st.json(input_data)

# Debug information
if st.checkbox("Show debug info"):
    st.subheader("Debug Information")
    
    if model:
        st.write("Model type:", type(model))
        st.write("Model parameters:", model.get_params())
        
        if hasattr(model, 'feature_importances_'):
            st.write("Feature importances:", model.feature_importances_)
    
    if 'features' in locals():
        st.write("Preprocessed features:", features)