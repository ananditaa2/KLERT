import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from models.cnn_model import BrainTumorCNN

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)


# ============================================================
# 2. CLASS NAMES
# MUST MATCH ImageFolder
# ============================================================

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ============================================================
# 3. LOAD MODEL
# ============================================================

model = BrainTumorCNN().to(device)

model_path = "brain_tumor_cnn_v2.pth"

print("\nLoading model from:")
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

print("✅ Model Loaded Successfully")


# ============================================================
# 4. IMAGE PATH
# USE EXACTLY THE SAME IMAGE AS predict.py
# ============================================================

image_path = "sample4.jpg"

print("\nLoading MRI:")
print(os.path.abspath(image_path))

if not os.path.exists(image_path):
    raise FileNotFoundError(
        f"Image not found: {os.path.abspath(image_path)}"
    )

original_image = Image.open(
    image_path
).convert("RGB")

print("Image Loaded Successfully!")
print("Image Size:", original_image.size)


# ============================================================
# 5. TRANSFORM
# IMPORTANT:
# This MUST be identical to training.
#
# Your training dataset currently uses:
# Resize -> ToTensor
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================================================
# 6. PREPARE IMAGE
# ============================================================

image_tensor = transform(
    original_image
)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device)

print(
    "\nInput Tensor Shape:",
    image_tensor.shape
)


# ============================================================
# 7. CNN PREDICTION
# ============================================================

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


# ============================================================
# 8. GET PREDICTED CLASS
# ============================================================

predicted_index = predicted.item()

predicted_class = class_names[predicted_index]

confidence_value = confidence.item() * 100


# ============================================================
# 9. PRINT PREDICTION
# ============================================================

print("\n======================================")
print("           CNN PREDICTION")
print("======================================")

print(
    "Prediction :",
    predicted_class
)

print(
    f"Confidence : {confidence_value:.2f}%"
)

print("======================================")


# ============================================================
# 10. PRINT ALL PROBABILITIES
# ============================================================

print("\nClass Probabilities:\n")

for i, probability in enumerate(
    probabilities[0]
):

    print(
        f"{class_names[i]:12s}: "
        f"{probability.item() * 100:.2f}%"
    )


# ============================================================
# 11. IMPORTANT DEBUG
# ============================================================

print("\nPredicted Class Index:", predicted_index)

print(
    "Predicted Class:",
    class_names[predicted_index]
)


# ============================================================
# 12. GRAD-CAM TARGET LAYER
#
# Your CNN:
#
# features[0]  Conv2d
# features[1]  BatchNorm
# features[2]  ReLU
# features[3]  MaxPool
#
# features[4]  Conv2d
# features[5]  BatchNorm
# features[6]  ReLU
# features[7]  MaxPool
#
# features[8]  Conv2d
# features[9]  BatchNorm
# features[10] ReLU
# features[11] MaxPool
#
# features[12] Conv2d  <-- LAST CONVOLUTION
# features[13] BatchNorm
# features[14] ReLU
# features[15] MaxPool
#
# ============================================================

target_layers = [
    model.features[12]
]


# ============================================================
# 13. CREATE GRAD-CAM
# ============================================================

cam = GradCAM(
    model=model,
    target_layers=target_layers
)


# ============================================================
# 14. GENERATE GRAD-CAM
#
# VERY IMPORTANT:
# Explicitly tell Grad-CAM to explain the
# predicted class.
# ============================================================

grayscale_cam = cam(
    input_tensor=image_tensor,
    targets=None
)

grayscale_cam = grayscale_cam[0]


# ============================================================
# 15. PREPARE ORIGINAL IMAGE
# ============================================================

display_image = original_image.resize(
    (224, 224)
)

display_image = np.array(
    display_image
).astype(np.float32) / 255.0


# ============================================================
# 16. CREATE HEATMAP
# ============================================================

visualization = show_cam_on_image(
    display_image,
    grayscale_cam,
    use_rgb=True
)


# ============================================================
# 17. SAVE RESULT
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

gradcam_path = (
    "results/gradcam_result.jpg"
)

Image.fromarray(
    visualization
).save(
    gradcam_path
)


# ============================================================
# 18. DISPLAY
# ============================================================

plt.figure(
    figsize=(8, 8)
)

plt.imshow(
    visualization
)

plt.title(
    f"Grad-CAM Explanation\n"
    f"Prediction: {predicted_class}\n"
    f"Confidence: {confidence_value:.2f}%"
)

plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# 19. FINAL OUTPUT
# ============================================================

print("\n======================================")
print("✅ Grad-CAM Completed Successfully!")
print("======================================")

print(
    "\nGrad-CAM saved to:"
)

print(
    os.path.abspath(
        gradcam_path
    )
)

print(
    "\nPrediction:",
    predicted_class
)

print(
    f"Confidence: {confidence_value:.2f}%"
)

print(
    "\nProcess finished successfully."
)