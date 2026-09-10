import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# -----------------------------------
# Page configuration
# -----------------------------------

st.set_page_config(
    page_title="Indian Confectionery Classifier",
    page_icon="🍬",
    layout="centered"
)

# -----------------------------------
# Title
# -----------------------------------

st.title("🍬 Indian Traditional Confectionery Classifier")

st.write(
    "Upload an image of an Indian traditional confectionery "
    "and the AI model will predict its name."
)

# -----------------------------------
# Load model
# -----------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "indian_confectionery_model.keras"
    )

model = load_model()

# -----------------------------------
# Load class names
# -----------------------------------

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# -----------------------------------
# Image upload
# -----------------------------------

uploaded_file = st.file_uploader(
    "📷 Upload a confectionery image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------
# Prediction
# -----------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Predict Confectionery"):

        # Resize image
        img = image.resize((224, 224))

        # Convert to NumPy array
        img_array = np.array(img)

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # Prediction
        predictions = model.predict(
            img_array,
            verbose=0
        )

        predicted_index = np.argmax(
            predictions[0]
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = (
            predictions[0][predicted_index] * 100
        )

        # -----------------------------------
        # Display result
        # -----------------------------------

        st.success(
            f"🍬 Prediction: {predicted_class}"
        )

        st.info(
            f"🎯 Confidence: {confidence:.2f}%"
        )

        # -----------------------------------
        # Show all predictions
        # -----------------------------------

        st.subheader("Prediction probabilities")

        for i, probability in enumerate(predictions[0]):

            st.write(
                f"{class_names[i]}: "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )
