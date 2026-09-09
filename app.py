import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from dataset import get_transforms
from model import build_model

CHECKPOINT_PATH = "./models/best_confectionery_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
class_names = checkpoint["class_names"]
model = build_model(num_classes=len(class_names), backbone=checkpoint.get("backbone", "efficientnet_b0"), pretrained=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

_, val_tf = get_transforms()

def classify_mithai(img):
    if img is None:
        return {}
    tensor = val_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
    return {class_names[i].replace('_', ' ').title(): float(probs[i]) for i in range(len(class_names))}

demo = gr.Interface(
    fn=classify_mithai,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Indian Confectionery (Mithai) Identifier",
    description="Upload an image of an Indian sweet (e.g., Gulab Jamun, Kaju Katli, Jalebi, Rasgulla) to classify it."
)

if __name__ == "__main__":
    demo.launch()
