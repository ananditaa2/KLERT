import os
import torch
from torchvision import transforms
from PIL import Image

from models.cnn_model import BrainTumorCNN


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)


# ============================================================
# Class Names
# IMPORTANT: Must match ImageFolder order
# ============================================================

class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


# ============================================================
# Load CNN Model
# ============================================================

model = BrainTumorCNN().to(device)

model_path = "brain_tumor_cnn_v2.pth"

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file not found: {os.path.abspath(model_path)}"
    )

print("Loading CNN model from:")
print(os.path.abspath(model_path))

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.eval()

print("CNN model loaded successfully!")


# ============================================================
# Image Transformation
# MUST match the transformation used during training
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Classification Function
# ============================================================

def classify_mri(image_path):

    # Check image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {os.path.abspath(image_path)}"
        )

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Transform image
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # CNN prediction
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

    # Convert result
    predicted_class = class_names[predicted.item()]

    confidence_value = confidence.item() * 100

    # Store probabilities
    class_probabilities = {}

    for i, probability in enumerate(probabilities[0]):

        class_probabilities[class_names[i]] = (
            probability.item() * 100
        )

    # Return structured result
    result = {

        "prediction": predicted_class,

        "confidence": confidence_value,

        "probabilities": class_probabilities,

        "image_path": os.path.abspath(image_path)

    }

    return result


# ============================================================
# Test Pipeline
# ============================================================

if __name__ == "__main__":

    image_path = ""

    result = classify_mri(image_path)

    print("\n===================================")
    print("       MRI CLASSIFICATION")
    print("===================================")

    print(
        "Prediction :",
        result["prediction"]
    )

    print(
        "Confidence :",
        f"{result['confidence']:.2f}%"
    )

    print("\nClass Probabilities:")

    for class_name, probability in result["probabilities"].items():

        print(
            f"{class_name:12s}: "
            f"{probability:.2f}%"
        )

    print("\n===================================")
    print("Classification Completed!")
    print("===================================")