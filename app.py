import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model("fruit_ripeness.keras")
classes = json.load(open("classes.json"))

st.title("Fruit Ripeness Grading Assistant")
f = st.file_uploader("Upload banana image", type=["jpg", "jpeg", "png"])
if f:
    img = Image.open(f).convert("RGB")
    st.image(img, width=300)
    x = np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), 0)
    p = model.predict(x)[0]
    st.success(f"Grade: {classes[int(p.argmax())]} ({p.max() * 100:.1f}% confidence)")
