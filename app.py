import streamlit as st
import torch
import json

from model import build_model


st.title("🍬 Indian Confectionery Classifier")

# Device
device = torch.device("cpu")

# Load checkpoint
checkpoint = torch.load(
    "best_confectionery_model.pth",
    map_location=device,
    weights_only=False
)

# Get information from checkpoint
class_names = checkpoint["class_names"]
backbone = checkpoint["backbone"]

# Build model
model = build_model(
    num_classes=len(class_names),
    backbone=backbone,
    pretrained=False
)

# Load trained weights
model.load_state_dict(checkpoint["model_state_dict"])

# Evaluation mode
model.to(device)
model.eval()

st.success("✅ PyTorch model loaded successfully!")

st.write("### Model Information")
st.write(f"Backbone: {backbone}")
st.write(f"Number of classes: {len(class_names)}")
st.write(f"Validation accuracy: {checkpoint['val_acc'] * 100:.2f}%")

st.write("### Classes")

for name in class_names:
    st.write(f"• {name}")
