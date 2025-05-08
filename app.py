import streamlit as st
import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Eyedentify", layout="centered")

# --- SIDEBAR ---
with st.sidebar:
    st.title("What is Eyedentify?")
    with st.expander("Click to Learn"):
        st.write("""
            *Eyedentify* is a smart computer vision tool that:
            - Captures live webcam feed
            - Detects faces in real-time
            - Uses a trained model to classify faces as *Real* or *Fake*
            - Supports uploading images and videos for visual inspection
        """)

# --- TITLE ---
st.markdown("<h1 style='text-align: center;'>👁 Eyedentify</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Real-Time Face Liveness Detection App</h4>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- LOAD MODEL ---
model = load_model("path_to_your_model.h5")  # Replace with your actual model path

# --- FACE DETECTOR ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- PREPROCESS FACE FOR MODEL ---
def preprocess_face(face_img):
    resized = cv2.resize(face_img, (128, 128))  # adjust based on your model
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

# --- CHOOSE MODE ---
option = st.radio("Choose an input method:", ["Use Webcam", "Upload Image", "Upload Video"], horizontal=True)

# --- WEBCAM MODE ---
if option == "Use Webcam":
    col1, col2 = st.columns([1, 2])
    with col1:
        start_cam = st.checkbox("Start Webcam")
    FRAME_WINDOW = col2.image([])

    camera = cv2.VideoCapture(0)

    while start_cam:
        ret, frame = camera.read()
        if not ret:
            st.warning("Unable to access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            try:
                input_face = preprocess_face(face_crop)
                pred = model.predict(input_face)[0][0]
                label = "Real" if pred > 0.5 else "Fake"
                color = (0, 255, 0) if label == "Real" else (0, 0, 255)
            except:
                label = "Unknown"
                color = (255, 255, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

    camera.release()

# --- IMAGE UPLOAD ---
elif option == "Upload Image":
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

# --- VIDEO UPLOAD ---
elif option == "Upload Video":
    vid_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())
        st.video(tfile.name)
        st.success("Video uploaded and ready to play.")

# --- FOOTER ---
st.markdown("<hr><p style='text-align: center;'>Made with ❤ by Your Name</p>", unsafe_allow_html=True)