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

print("\nLoading model:")
print(os.path.abspath(model_path))

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.eval()

print("Model Loaded Successfully!")


# ============================================================
# 4. IMAGE PATH
# ============================================================

image_path = "sample4.jpg"

print("\nLoading image:")
print(os.path.abspath(image_path))

original_image = Image.open(
    image_path
).convert("RGB")

print("Image size:", original_image.size)


# ============================================================
# 5. TRANSFORM
# MUST MATCH TRAINING
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
    "Input tensor shape:",
    image_tensor.shape
)


# ============================================================
# 7. PREDICTION
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


predicted_index = predicted.item()

predicted_class = class_names[predicted_index]

confidence_value = confidence.item() * 100


# ============================================================
# 8. PRINT RESULT
# ============================================================

print("\n======================================")
print("CNN PREDICTION")
print("======================================")

print(
    "Prediction:",
    predicted_class
)

print(
    f"Confidence: {confidence_value:.2f}%"
)

print("======================================")


# ============================================================
# 9. TARGET LAYER
# LAST CONVOLUTIONAL LAYER
# ============================================================

target_layers = [
    model.features[12]
]


# ============================================================
# 10. CREATE GRAD-CAM
# ============================================================

cam = GradCAM(
    model=model,
    target_layers=target_layers
)


# ============================================================
# 11. GENERATE HEATMAP
# ============================================================

grayscale_cam = cam(
    input_tensor=image_tensor
)

grayscale_cam = grayscale_cam[0]


# ============================================================
# 12. PREPARE DISPLAY IMAGE
# ============================================================

display_image = original_image.resize(
    (224, 224)
)

display_image = np.array(
    display_image
).astype(
    np.float32
) / 255.0


# ============================================================
# 13. CREATE OVERLAY
# ============================================================

visualization = show_cam_on_image(
    display_image,
    grayscale_cam,
    use_rgb=True
)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


# Original
plt.figure(figsize=(6, 6))

plt.imshow(display_image)

plt.title("Original MRI")

plt.axis("off")

plt.tight_layout()

plt.savefig(
    "results/original_mri.jpg",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Grad-CAM heatmap
plt.figure(figsize=(6, 6))

plt.imshow(
    grayscale_cam,
    cmap="jet"
)

plt.title(
    f"Grad-CAM Heatmap\n"
    f"Prediction: {predicted_class}"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    "results/gradcam_heatmap.jpg",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Overlay
plt.figure(figsize=(6, 6))

plt.imshow(visualization)

plt.title(
    f"Grad-CAM Explanation\n"
    f"Prediction: {predicted_class}\n"
    f"Confidence: {confidence_value:.2f}%"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    "results/gradcam_overlay.jpg",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. CREATE COMBINED FIGURE
# ============================================================

plt.figure(figsize=(15, 5))


# Original
plt.subplot(1, 3, 1)

plt.imshow(display_image)

plt.title("Original MRI")

plt.axis("off")


# Heatmap
plt.subplot(1, 3, 2)

plt.imshow(
    grayscale_cam,
    cmap="jet"
)

plt.title("Grad-CAM Heatmap")

plt.axis("off")


# Overlay
plt.subplot(1, 3, 3)

plt.imshow(visualization)

plt.title(
    f"Grad-CAM Overlay\n"
    f"{predicted_class} ({confidence_value:.2f}%)"
)

plt.axis("off")


plt.tight_layout()

plt.savefig(
    "results/gradcam_analysis.jpg",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 16. FINAL OUTPUT
# ============================================================

print("\n======================================")
print("Grad-CAM Analysis Completed!")
print("======================================")

print("\nPrediction:", predicted_class)

print(
    f"Confidence: {confidence_value:.2f}%"
)

print("\nGenerated files:")

print(
    "results/original_mri.jpg"
)

print(
    "results/gradcam_heatmap.jpg"
)

print(
    "results/gradcam_overlay.jpg"
)

print(
    "results/gradcam_analysis.jpg"
)

print("\nDone!")