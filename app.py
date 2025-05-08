import tensorflow as tf # type: ignore
import pickle
from detector.preprocess import preprocess_input
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import random

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()
def predict_image(image_np):
    processed = preprocess_input(image_np)

    processed = processed.reshape(1, -1) if len(processed.shape) == 1 else processed
    processed_scaled = scaler.transform(processed)

    prediction = model.predict(processed_scaled)
    predicted_label = "Real" if prediction[0][0] > 0.5 else "Fake"
    confidence = round(float(prediction[0][0] * 100), 2)

    st.success(f"Prediction: {predicted_label} Face")
    st.metric("Confidence", f"{confidence}%" if predicted_label == "Real" else f"{100 - confidence}%")

st.set_page_config(page_title="EyeDentify", layout="centered")

# Sidebar
st.sidebar.title("EyeDentify")
st.sidebar.markdown("*Face Spoof Detection Demo*")
st.sidebar.markdown("Built with OpenCV + Streamlit")
# You can add a logo like this if you have one:
# st.sidebar.image("logo.png", use_column_width=True)

# Title
st.title(":eyes: EyeDentify - Face Spoof Detection")
st.markdown("<hr style='border:1px solid gray'/>", unsafe_allow_html=True)

# Description
with st.expander("What is EyeDentify?"):
    st.write("""
    EyeDentify is a face spoof detection app designed to distinguish real faces from spoofed ones (like photos, videos, or masks). 
    It uses image analysis and AI to make predictions.
    """)

# Input Method
input_type = st.radio("Choose input method:", ["Upload Image", "Use Webcam"])

# Fake prediction function (for demo)
def predict_placeholder(image):
    result = random.choice(["Real", "Fake"])
    confidence = round(random.uniform(70, 99), 2)
    
    st.success(f"Prediction: {result} Face")
    st.metric("Confidence", f"{confidence}%")

# Upload Image Flow
if input_type == "Upload Image":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col2:
            if st.button("Predict"):
                with st.spinner("Analyzing..."):
                    predict_placeholder(image)
    else:
        st.info("Please upload an image to proceed.")

# Webcam Flow
elif input_type == "Use Webcam":
    st.info("Click 'Capture' to take a frame from your webcam.")
    
    if st.button("Capture"):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Captured Frame", use_column_width=True)

            if st.button("Predict on Captured Image"):
                with st.spinner("Analyzing..."):
                    predict_placeholder(frame_rgb)
        else:
            st.error("Could not access webcam. Make sure it’s connected and not used by another app.")

# Footer
st.markdown("---")
st.markdown("Made with ❤ by Diyana and Nalin", unsafe_allow_html=True)