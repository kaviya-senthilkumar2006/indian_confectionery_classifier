import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Indian Confectionery Classifier",
    page_icon="🍬",
    layout="centered"
)

st.title("🍬 Indian Confectionery (Mithai) Identifier")
st.write("Upload an image of an Indian sweet (Gulab Jamun, Jalebi, Kaju Katli, Rasgulla, etc.) to identify it.")

# --- Model Architecture ---
def build_model(num_classes: int, backbone: str = "efficientnet_b0"):
    if backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model

# --- Cached Model Loader ---
@st.cache_resource
def load_classifier(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        return None, None

    device = torch.device("cpu")  # Streamlit Cloud runs on CPU
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    class_names = checkpoint.get("class_names", [
        "gulab_jamun", "jalebi", "kaju_katli", 
        "motichoor_ladoo", "peda", "rasgulla", "rasmalai"
    ])
    backbone = checkpoint.get("backbone", "efficientnet_b0")

    model = build_model(num_classes=len(class_names), backbone=backbone)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    return model, class_names, val_tf

# --- Load Model with Graceful Fallback ---
MODEL_PATH = "models/best_confectionery_model.pth"

with st.spinner("Loading AI model..."):
    loaded = load_classifier(MODEL_PATH)

if loaded is None or loaded[0] is None:
    st.warning(
        f"⚠️ Model checkpoint not found at `{MODEL_PATH}`.\n\n"
        "If you deployed to Streamlit Cloud, make sure your `.pth` model file is uploaded, "
        "or host it on Hugging Face / Google Drive and download it upon startup."
    )
    st.stop()  # Stops cleanly instead of showing a blank screen!

model, class_names, transform = loaded

# --- Image Upload Section ---
uploaded_file = st.file_uploader("Choose an image of a sweet...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Sweet", use_container_width=True)

    # Predict Button
    if st.button("🔍 Identify Confectionery", type="primary"):
        with st.spinner("Analyzing sweets..."):
            img_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)[0]

            top_probs, top_indices = torch.topk(probs, k=min(3, len(class_names)))

        st.subheader("Results:")
        for prob, idx in zip(top_probs, top_indices):
            name = class_names[idx.item()].replace("_", " ").title()
            confidence = prob.item() * 100
            st.write(f"**{name}**: {confidence:.1f}%")
            st.progress(float(prob.item()))
