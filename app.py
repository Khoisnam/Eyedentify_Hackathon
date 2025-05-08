import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import tempfile
import os


st.set_page_config(page_title="Eyedentify", layout="centered")


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


st.markdown("<h1 style='text-align: center;'>👁 Eyedentify</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Real-Time Face Liveness Detection App</h4>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


model = load_model("model.keras") 
model.summary() 


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def preprocess_face(face_img):
    resized = cv2.resize(face_img, (300, 300))  
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)


option = st.radio("Choose an input method:", ["Use Webcam", "Upload Image", "Upload Video"], horizontal=True)


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
        if face_crop.size == 0:
            continue  

        try:
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)  
            input_face = preprocess_face(face_crop)

            
            if input_face.shape[1:] != (300, 300, 3): 
                raise ValueError(f"Incorrect input shape: {input_face.shape}")

            pred = model.predict(input_face)[0][0]
            label = "Real" if pred > 0.5 else "Fake"
            color = (0, 255, 0) if label == "Real" else (0, 0, 255)

        except Exception as e:
            label = "Unknown"
            color = (255, 255, 0)
            print(f"Prediction failed: {e}")

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

    camera.release()

#image upload
elif option == "Upload Image":
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

#video upload
elif option == "Upload Video":
    vid_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())
        st.video(tfile.name)
        st.success("Video uploaded and ready to play.")

#footer
st.markdown("<hr><p style='text-align: center;'>Made with ❤ by Diyana and Nalin</p>", unsafe_allow_html=True)