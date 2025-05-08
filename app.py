import streamlit as st # type: ignore
from PIL import Image
import numpy as np # type: ignore

st.set_page_config(page_title="EyeDentify", layout="centered")
st.title("EyeDentify - Face Spoof Detection")

st.markdown("<hr style='border:1px solid gray'/>", unsafe_allow_html=True)

input_type = st.radio("Choose input method:", ["Upload Image", "Use Webcam"])

if input_type == "Upload Image":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Running spoof detection..."):
                # Placeholder: replace with real model prediction
                st.success("Prediction: Real Face (placeholder)")
    else:
        st.info("Please upload an image to proceed.")

elif input_type == "Use Webcam":
    st.warning("Webcam feature will be enabled once the model is integrated.")
    # Future use:
    # import cv2
    # cap = cv2.VideoCapture(0)
    # ret, frame = cap.read()
    # st.image(frame, channels="BGR")
    # cap.release()