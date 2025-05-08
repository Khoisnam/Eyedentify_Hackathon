import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Eyedentify", layout="centered")

# --- SIDEBAR ---
with st.sidebar:
    st.title("What is Eyedentify?")
    with st.expander("Click to Learn"):
        st.write("""
            *Eyedentify* is a smart computer vision app that:
            - Shows live webcam feed
            - Captures images or records video
            - Supports uploading images and videos
            Perfect for real-time identity capture, analysis, and demo purposes!
        """)

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>👁 Eyedentify</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Smart webcam tool for image & video processing</h4>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- SELECTION ---
option = st.radio("Choose an input method:", ["Use Webcam", "Upload Image", "Upload Video"], horizontal=True)

# Initialize camera
def init_camera():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        st.error("Unable to access webcam.")
        return None
    return cam

# Webcam Mode
if option == "Use Webcam":
    col1, col2, col3 = st.columns(3)
    with col1:
        show_cam = st.checkbox("Show Camera")
    with col2:
        capture_image = st.button("Capture Image")
    with col3:
        record_video = st.checkbox("Record Video")

    FRAME_WINDOW = st.image([])

    camera = init_camera()
    if camera and show_cam:
        if record_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            fps = 20.0
            width = int(camera.get(3))
            height = int(camera.get(4))
            out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
            st.success("Recording started...")

        start_time = time.time()
        while show_cam:
            ret, frame = camera.read()
            if not ret:
                st.warning("Webcam feed unavailable.")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb, channels="RGB")

            if capture_image:
                img_name = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(img_name, frame)
                st.success(f"Image saved as {img_name}")
                break

            if record_video:
                out.write(frame)

            # Optional timeout to prevent infinite loop
            if time.time() - start_time > 20 and not record_video:
                break

        camera.release()
        if record_video:
            out.release()
            st.success(f"Video saved as {video_filename}")

# Image Upload
elif option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

# Video Upload
elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        st.video(tfile.name)
        st.success("Video uploaded and ready to play.")

# --- FOOTER ---
st.markdown("<hr><p style='text-align: center;'>Made with ❤ by Your Name</p>", unsafe_allow_html=True)