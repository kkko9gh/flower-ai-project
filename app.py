
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from collections import Counter

st.set_page_config(page_title="Flower AI", page_icon="🌸")

@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model("flower_classifier.keras")
    with open("class_names.json", "r") as f:
        labels = json.load(f)
    return model, labels

model, labels = load_model_and_labels()

st.title("🌸 AI Flower Recognition")
st.write("Recognize flowers and analyze their colors.")

photo = st.camera_input("Take a flower photo")
if photo is None:
    photo = st.file_uploader(
        "Or upload a flower image",
        type=["jpg", "jpeg", "png"]
    )

if photo is not None:
    image = Image.open(photo).convert("RGB")
    st.image(image, caption="Your flower", use_container_width=True)

    # Prepare the image for the trained model
    resized = image.resize((160, 160))
    array = np.array(resized, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    # Predict flower category
    predictions = model.predict(array, verbose=0)[0]
    index = int(np.argmax(predictions))
    name = labels[index]
    confidence = float(predictions[index]) * 100

    st.subheader("🌼 Recognition Result")
    st.success(f"Flower category: {name}")
    st.metric("Model confidence", f"{confidence:.2f}%")

    # Estimate dominant colors from the whole image
    small = image.resize((100, 100))
    quantized = small.quantize(colors=3)
    palette = quantized.getpalette()
    color_counts = sorted(
        quantized.getcolors(10000),
        reverse=True
    )

    st.subheader("🎨 Dominant Image Colors")
    for count, color_index in color_counts[:3]:
        i = color_index * 3
        rgb = tuple(palette[i:i + 3])
        hex_color = "#{:02X}{:02X}{:02X}".format(*rgb)

        st.markdown(
            f"<div style='background:{hex_color};"
            f"padding:12px;border-radius:8px;color:"
            f"{'#000' if sum(rgb) > 380 else '#fff'}'>"
            f"{hex_color} — {count / 100:.1f}% of image pixels"
            f"</div>",
            unsafe_allow_html=True
        )

    st.caption(
        "Color analysis includes the background and leaves. "
        "Predictions may be incorrect for unfamiliar images."
    )
