import os

import torch
from torchvision import transforms
from PIL import Image

from models.cnn_model import BrainTumorCNN


# =====================================================
# Device
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)


# =====================================================
# Class Names
# MUST MATCH ImageFolder ORDER
# =====================================================

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# =====================================================
# Model
# =====================================================

model = BrainTumorCNN().to(device)

model_path = "brain_tumor_cnn_v2.pth"

print("Loading model from:")
print(os.path.abspath(model_path))

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model not found: {os.path.abspath(model_path)}"
    )

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.eval()


# =====================================================
# Transform
# MUST BE SAME AS TRAINING
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# =====================================================
# Image
# =====================================================

image_path = "sampleee.jpg"

print("\nImage path:")
print(os.path.abspath(image_path))

if not os.path.exists(image_path):
    raise FileNotFoundError(
        f"Image not found: {os.path.abspath(image_path)}"
    )


image = Image.open(
    image_path
).convert("RGB")


print("Image size:", image.size)


# =====================================================
# Preprocessing
# =====================================================

image_tensor = transform(image)

image_tensor = image_tensor.unsqueeze(0).to(device)


# =====================================================
# Prediction
# =====================================================

with torch.no_grad():

    outputs = model(image_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, predicted = torch.max(
        probabilities,
        dim=1
    )


# =====================================================
# Result
# =====================================================

predicted_index = predicted.item()

print("\n==============================")

print(
    "Prediction:",
    class_names[predicted_index]
)

print(
    "Confidence:",
    f"{confidence.item() * 100:.2f}%"
)

print("==============================")


# =====================================================
# All Probabilities
# =====================================================

print("\nClass Probabilities:")

for i, probability in enumerate(
    probabilities[0]
):

    print(
        f"{class_names[i]:12s}: "
        f"{probability.item() * 100:.2f}%"
    )


print("\nPrediction Complete.")