import json
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from dataset import get_transforms
from model import build_model

def predict(image_path: str, checkpoint_path: str, top_k: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    backbone = checkpoint.get("backbone", "efficientnet_b0")

    model = build_model(num_classes=len(class_names), backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _, val_tf = get_transforms()
    image = Image.open(image_path).convert("RGB")
    tensor = val_tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(class_names)))

    print(f"\nPredictions for: {image_path}")
    print("-" * 35)
    for i in range(top_probs.size(0)):
        class_name = class_names[top_indices[i].item()]
        confidence = top_probs[i].item() * 100
        print(f"{i+1}. {class_name.replace('_', ' ').title()}: {confidence:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--checkpoint", type=str, default="./models/best_confectionery_model.pth")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    predict(args.image, args.checkpoint, args.top_k)
