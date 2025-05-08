import streamlit as st # type: ignore
from PIL import Image
st.title("EyeDentify:AI based face detection")

input_type=st.radio("Choose input method:",["upload image","use webcam"])
if input_type=="Upload image":
    uploaded_file=st.file_uploader("Upload a face image",type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        image=Image.open(uploaded_file)
        st.image(image,caption="Uploaded image",use_column_width=True)
        st.success("Image loaded.Waiting for model...")
elif input_type=="Use webcam":
    st.warning("webcam feature will be enabled once the model is integrated") 
        