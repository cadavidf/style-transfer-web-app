import numpy as np
import streamlit as st
from PIL import Image
import cv2
import imutils
from neural_style_transfer import get_model_from_path, style_transfer
from data import style_models_dict, content_images_dict, content_images_name

def image_input(style_model_name):
    style_model_path = style_models_dict[style_model_name]

    # Add error handling for model loading
    try:
        model = get_model_from_path(style_model_path)
    except RuntimeError as e:
        st.error(f"Failed to load the model from {style_model_path}: {e}")
        return

    if st.sidebar.checkbox('Upload'):
        content_file = st.sidebar.file_uploader("Choose a Content Image", type=["png", "jpg", "jpeg"])
    else:
        content_name = st.sidebar.selectbox("Choose the content images:", content_images_name)
        content_file = content_images_dict[content_name]

    if content_file is not None:
        content = Image.open(content_file)
        content = np.array(content)  # Convert PIL image to OpenCV image
        content = cv2.cvtColor(content, cv2.COLOR_RGB2BGR)
    else:
        st.warning("Upload an Image OR Untick the Upload Button")
        st.stop()

    WIDTH = st.sidebar.select_slider('QUALITY (May reduce the speed)', list(range(150, 501, 50)), value=200)
    content = imutils.resize(content, width=WIDTH)
    generated = style_transfer(content, model)
    st.sidebar.image(content, width=300, channels='BGR')
    st.image(generated, channels='BGR', clamp=True)
