import streamlit as st
from PIL import Image
import numpy as np
import cv2 # type: ignore

st.set_page_config(page_title="EyeDentify", layout="centered")
st.title("EyeDentify - Face Spoof Detection")

st.markdown("<hr style='border:1px solid gray'/>", unsafe_allow_html=True)

input_type = st.radio("Choose input method:", ["Upload Image", "Use Webcam"])

def predict_placeholder(image):
    # Convert image to numpy if it's a PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Placeholder model simulation
    st.success("Prediction: Real Face (placeholder)")

if input_type == "Upload Image":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Running spoof detection..."):
                predict_placeholder(image)
    else:
        st.info("Please upload an image to proceed.")

elif input_type == "Use Webcam":
    st.info("Click 'Capture' to take a frame from your webcam.")
    if st.button("Capture"):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame, caption="Captured Frame", channels="RGB", use_column_width=True)

            if st.button("Predict on Captured Image"):
                with st.spinner("Running spoof detection..."):
                    predict_placeholder(frame)
        else:
            st.error("Failed to access webcam.")