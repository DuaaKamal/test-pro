import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =============================

# =============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Mai94.keras")

model = load_model()
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# =============================

# =============================
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: white; background-color: #1E3A8A;
    padding: 15px; border-radius: 10px;'> Brain Tumor Detection from MRI images </h1>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    image_input = Image.open(uploaded_file).convert("RGB")
    img_resized = image_input.resize((224, 224))   # نفس Jupyter
    img_array = np.array(img_resized, dtype=np.float32)  # بدون تقسيم /255
    img_array = np.expand_dims(img_array, axis=0)        # batch

   
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100
    predicted_label = class_labels[predicted_index]

   
    st.image(image_input, caption="Uploaded MRI Image", use_container_width=True)
    st.success(f" Prediction: {predicted_label}")
    st.info(f" Confidence: {confidence:.2f}%")