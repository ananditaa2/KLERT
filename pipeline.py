import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from models.cnn_model import BrainTumorCNN

# ============================================================
# Device Setup
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ============================================================
# Class Names
# ============================================================
class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# ============================================================
# 1. Custom BrainTumorCNN Model
# ============================================================
cnn_model = BrainTumorCNN().to(device)
cnn_path = "brain_tumor_cnn_v2.pth"
if os.path.exists(cnn_path):
    print(f"Loading Custom CNN model from: {os.path.abspath(cnn_path)}")
    cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
cnn_model.eval()

# ============================================================
# 2. ResNet18 Pretrained Model
# ============================================================
resnet_model = models.resnet18(weights=None)
resnet_model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(resnet_model.fc.in_features, 4)
)
resnet_model = resnet_model.to(device)
resnet_path = "resnet18_brain_tumor.pth"
if os.path.exists(resnet_path):
    print(f"Loading ResNet18 model from: {os.path.abspath(resnet_path)}")
    resnet_model.load_state_dict(torch.load(resnet_path, map_location=device))
resnet_model.eval()

# ============================================================
# 3. VGG16 Pretrained Model
# ============================================================
vgg_model = models.vgg16(weights=None)
vgg_model.classifier[6] = nn.Linear(vgg_model.classifier[6].in_features, 4)
vgg_model = vgg_model.to(device)
vgg_path = "vgg16_brain_tumor.pth"
if os.path.exists(vgg_path):
    print(f"Loading VGG16 model from: {os.path.abspath(vgg_path)}")
    vgg_model.load_state_dict(torch.load(vgg_path, map_location=device))
vgg_model.eval()

# Models Dictionary mapping model key -> PyTorch Module
models_dict = {
    "cnn": cnn_model,
    "resnet": resnet_model,
    "vgg": vgg_model
}

# Image Preprocessing Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ============================================================
# Classification Function
# ============================================================
def classify_mri(image_path: str, model_type: str = "cnn"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {os.path.abspath(image_path)}")

    model_type_clean = model_type.lower().strip()
    if model_type_clean not in models_dict:
        model_type_clean = "cnn"

    target_model = models_dict[model_type_clean]
    target_model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = target_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted = torch.max(probabilities, dim=0)

    predicted_index = predicted.item()
    predicted_class = class_names[predicted_index]
    confidence_value = confidence.item() * 100.0

    class_probabilities = {}
    for i, prob in enumerate(probabilities):
        class_probabilities[class_names[i]] = round(prob.item() * 100.0, 2)

    return {
        "prediction": predicted_class,
        "confidence": round(confidence_value, 2),
        "probabilities": class_probabilities,
        "model_used": model_type_clean,
        "image_path": os.path.abspath(image_path)
    }

if __name__ == "__main__":
    if os.path.exists("sample4.jpg"):
        res = classify_mri("sample4.jpg", "cnn")
        print("Test Classification Result:", res)