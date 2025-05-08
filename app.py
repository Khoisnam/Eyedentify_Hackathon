import streamlit as st # type: ignore
from PIL import Image # type: ignore
import numpy as np # type: ignore
import cv2 # type: ignore
import pickle
import time
import os
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
import queue
import av # type: ignore
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase # type: ignore

st.set_page_config(page_title="EyeDentify", layout="centered")

# Load the face detector model
def load_face_detector():
    protoPath = os.path.sep.join(["detector", "deploy.prototxt"])
    modelPath = os.path.sep.join(["detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    return detector

# Load the liveness detection model
def load_liveness_model():
    model = load_model("liveness_model_v3.keras")
    le = pickle.loads(open("le_model_v3.pickle", "rb").read())
    return model, le

# Initialize models
try:
    face_detector = load_face_detector()
    liveness_model, label_encoder = load_liveness_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading models: {e}")

st.sidebar.title("EyeDentify")
st.sidebar.markdown("Face Spoof Detection Demo")
st.sidebar.markdown("Built with OpenCV + Streamlit + TensorFlow")

with st.sidebar.expander("About Model"):
    st.markdown("""
    #### Liveness Detection Model

    Architecture: Custom CNN (LivenessNet)

    Input Size: 32x32 pixels

    Output Classes: Real, Fake
    """)

st.title(":eyes: EyeDentify - Face Spoof Detection")
st.markdown("<hr style='border:1px solid gray'/>", unsafe_allow_html=True)

with st.expander("What is EyeDentify?"):
    st.write("""
    EyeDentify detects whether a face is real or fake using deep learning.
    It detects the face and then classifies it as live or spoofed.
    """)

def process_frame(frame):
    results = []
    if not model_loaded:
        return frame, results

    image_cv = frame.copy()
    result_image = image_cv.copy()

    h, w = result_image.shape[:2]

    blob = cv2.dnn.blobFromImage(image_cv, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()
    print("Detections:",detections.shape[2])
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)

            face = image_cv[startY:endY, startX:endX]
            if face.size == 0:
                continue

            try:
                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                preds = liveness_model.predict(face)[0]
                j = np.argmax(preds)
                label_name = label_encoder.classes_[j]
                probability = preds[j] * 100

                results.append({
                    "label": label_name,
                    "confidence": probability,
                    "position": (startX, startY, endX, endY)
                })

                label_text = f"{label_name}: {probability:.2f}%"
                color = (0, 255, 0) if label_name == "real" else (0, 0, 255)
                cv2.putText(result_image, label_text, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(result_image, (startX, startY), (endX, endY), color, 2)
            except Exception as e:
                st.warning(f"Error processing face: {e}")
                continue
            except Exception as e:
                st.warning(f"Error processing face: {e}")
                continue

    
    return result_image, results

class LivenessDetectionProcessor(VideoProcessorBase):
    def _init_(self):
        self.results_queue = queue.Queue(maxsize=1)

    def recv(self, frame):
        print("Processing frame")

        img = frame.to_ndarray(format="bgr24")
        result_img, results = process_frame(img)

        try:
            while not self.results_queue.empty():
                self.results_queue.get_nowait()
            self.results_queue.put(results, block=False)
        except queue.Full:
            pass

        return av.VideoFrame.from_ndarray(result_img, format="bgr24")

    def get_latest_results(self):
        if not self.results_queue.empty():
            return self.results_queue.get_nowait()
        return []

input_type = st.radio("Choose input method:", ["Real-time Webcam", "Upload Image"])

if input_type == "Real-time Webcam":
    st.info("Click 'Start' to begin real-time liveness detection through your webcam.")

    webrtc_ctx = webrtc_streamer(
        key="liveness-detection",
        video_processor_factory=LivenessDetectionProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    results_placeholder = st.empty()

    if webrtc_ctx.video_processor:
        while webrtc_ctx.state.playing:
            results = webrtc_ctx.video_processor.get_latest_results()
            if results:
                with results_placeholder.container():
                    col1, col2 = st.columns(2)
                    for idx, result in enumerate(results):
                        col = col1 if idx % 2 == 0 else col2
                        with col:
                            if result['label'] == 'real':
                                st.success(f"REAL FACE detected with {result['confidence']:.2f}% confidence")
                            else:
                                st.error(f"FAKE FACE detected with {result['confidence']:.2f}% confidence")
            time.sleep(0.1)

elif input_type == "Upload Image":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col2:
            if st.button("Predict"):
                with st.spinner("Analyzing..."):
                    result_image, results = process_frame(image_np)
                    if results:
                        st.image(result_image, caption="Analysis Result", use_column_width=True)
                        for idx, result in enumerate(results):
                            with st.expander(f"Face #{idx + 1} - {result['label'].upper()}"):
                                if result['label'] == 'real':
                                    st.success(f"REAL FACE detected with {result['confidence']:.2f}% confidence")
                                else:
                                    st.error(f"FAKE FACE detected with {result['confidence']:.2f}% confidence")
                    else:
                        st.warning("No faces detected in the image. Please try another image.")
    else:
        st.info("Please upload an image to proceed.")

with st.expander("Tips for Better Results"):
    st.markdown("""
    - Ensure good, even lighting on your face
    - Position your face clearly in view of the camera
    - Try showing a photo or screen-based face for spoof testing
    """)

st.markdown("---")
st.markdown("Made with ❤ by Diyana and Nalin")