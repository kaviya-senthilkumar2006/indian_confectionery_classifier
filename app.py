import os
import json
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# 1. Page Configuration
st.set_page_config(
    page_title="Indian Confectionery Identifier",
    page_icon="🍬",
    layout="centered"
)

st.title("🍬 Indian Confectionery (Mithai) Identifier")
st.write("Upload a photo of an Indian sweet (e.g., Gulab Jamun, Jalebi, Kaju Katli, Rasgulla, etc.) to identify it.")

# 2. Confectionery Classes
CLASSES = [
    "gulab_jamun",
    "jalebi",
    "kaju_katli",
    "motichoor_ladoo",
    "peda",
    "rasgulla",
    "rasmalai",
    "mysore_pak",
    "gujiya"
]

# 3. Model Architecture & Cache
def build_model(num_classes: int):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    return model

@st.cache_resource
def load_model():
    model_path = "models/best_confectionery_model.pth"
    model = build_model(len(CLASSES))
    device = torch.device("cpu")
    
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model_loaded = True
        except Exception as e:
            model_loaded = False
    else:
        model_loaded = False
        
    model.to(device)
    model.eval()
    return model, model_loaded

with st.spinner("Initializing classifier..."):
    model, is_trained = load_model()

if not is_trained:
    st.info("ℹ️ Running in demo mode (model weights `.pth` not found in `models/` directory). Upload an image to test the UI.")

# 4. Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. UI: File Uploader
uploaded_file = st.file_uploader("Upload an image of a sweet...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("Predictions")
        with st.spinner("Analyzing sweet..."):
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)[0]
                
            top_probs, top_indices = torch.topk(probs, k=min(3, len(CLASSES)))

            for prob, idx in zip(top_probs, top_indices):
                sweet_name = CLASSES[idx.item()].replace("_", " ").title()
                confidence = float(prob.item()) * 100
                st.write(f"**{sweet_name}** ({confidence:.1f}%)")
                st.progress(float(prob.item()))
