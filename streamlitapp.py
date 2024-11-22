import streamlit as st
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')
print(model.input_shape)

def predict_price(features):
    features = np.array(features).reshape((1, 7))
    prediction = model.predict(features)
    category = np.argmax(prediction)
    
    # Map category number to price range label
    categories = {
        0: "Low-cost",
        1: "Medium-cost",
        2: "High-cost",
        3: "Very high-cost"
    }
    return categories.get(category, "Unknown category")

# Streamlit app layout
st.title("Mobile Phone Price Prediction")
st.write("Provide the following details to predict the mobile phone price category.")

# Define input fields for the 7 features used in model training
battery_power = st.number_input("Battery Power (mAh)", min_value=500, max_value=5000, step=50)
ram = st.number_input("RAM (MB)", min_value=256, max_value=8192, step=256)
px_height = st.number_input("Pixel Resolution Height", min_value=0, max_value=2000, step=100)
px_width = st.number_input("Pixel Resolution Width", min_value=0, max_value=2000, step=100)
int_memory = st.number_input("Internal Memory (GB)", min_value=2, max_value=128, step=1)
clock_speed = st.number_input("Clock Speed (GHz)", min_value=0.5, max_value=3.5, step=0.1)
talk_time = st.number_input("Talk Time (hours)", min_value=2, max_value=30, step=1)

# Collect input data
input_data = [battery_power, ram, px_height, px_width, int_memory, clock_speed, talk_time]

if st.button("Predict Price Category"):
    prediction = predict_price(input_data)
    st.write(f"Predicted Price Category: {prediction}")
