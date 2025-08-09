import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('rs_model.pkl')

# Page title and description
st.set_page_config(page_title="Restaurant Menu Optimization", page_icon="🍽️")
st.title('Restaurant Menu Optimization')
st.markdown("""
This application predicts the profitability of menu items based on their category and price. 
Please enter the details below to get the prediction.
""")

# Sidebar for user inputs
st.sidebar.header('User Input Features')

# Function to get user input
def user_input_features():
    restaurant_id = st.sidebar.selectbox('Restaurant ID', ['R001','R002','R003'])
    menu_category = st.sidebar.selectbox('Menu Category', ['Appetizer', 'Beverages', 'Dessert', 'Main Course'])
    menu_item = st.sidebar.selectbox('Menu Item', ['Bruschetta','Caprese Salad','Spinach Artichoke Dip','Stuffed Mushrooms',
                                                   'Coffee','Iced Tea','Lemonade','Soda',
                                                   'Chocolate Lava Cake','Fruit Tart','New York Cheesecake','Tiramisu',
                                                   'Chicken Alfredo','Grilled Steak','Shrimp Scampi','Vegetable Stir-Fry'])
    price = st.sidebar.number_input('Price ($)', min_value=0.0, value=10.0, step=0.5, format='%f')
    data = {
        'RestaurantID' : restaurant_id,
        'MenuCategory': menu_category,
        'MenuItem' : menu_item,
        'Price': price
    }
    return data

# Get user input
input_data = user_input_features()

# Convert input data to required format
restoID = {'R001': 0, 'R002': 1, 'R003': 2}
categories = {'Appetizer': 0, 'Beverages' : 1, 'Dessert': 2, 'Main Course': 3}
items = {'Bruschetta': 0,'Caprese Salad': 1,'Spinach Artichoke Dip': 2,'Stuffed Mushrooms': 3,
        'Coffee': 4,'Iced Tea': 5,'Lemonade': 6,'Soda': 7,
        'Chocolate Lava Cake': 8,'Fruit Tart': 9,'New York Cheesecake' :10,'Tiramisu': 11,
        'Chicken Alfredo': 12,'Grilled Steak': 13,'Shrimp Scampi': 14,'Vegetable Stir-Fry': 15}

restaurant_id = restoID[input_data['RestaurantID']]
menu_category = categories[input_data['MenuCategory']]
menu_item = items[input_data['MenuItem']]
price = input_data['Price']
features = np.array([restaurant_id, menu_category, menu_item, price]).reshape(1, -1)

# Predict button
if st.sidebar.button('Predict Profitability'):
    prediction = model.predict(features)
    if prediction[0] == 0:
        result = "Low Profit"
    elif prediction[0] == 1:
        result = "Medium Profit"
    else:
        result = "High Profit"
    st.subheader('Prediction Result')
    st.write(f'The predicted profitability of the item is: **{result}**')

# Display input data
st.subheader('Input Data')
st.write('Restaurant ID:', input_data['RestaurantID'])
st.write('Menu Category:', input_data['MenuCategory'])
st.write('Menu Item:', input_data['MenuItem'])
st.write('Price ($):', input_data['Price'])