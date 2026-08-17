import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from models.cnn_model import BrainTumorCNN

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)


# ============================================================
# 2. CLASS NAMES
# MUST MATCH ImageFolder CLASS ORDER
# ============================================================

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

model = BrainTumorCNN().to(device)

model_path = "brain_tumor_cnn_v2.pth"

print("\nLoading model from:")
print(os.path.abspath(model_path))

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model not found:\n{os.path.abspath(model_path)}"
    )

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

print("\nLoading MRI:")
print(os.path.abspath(image_path))

if not os.path.exists(image_path):
    raise FileNotFoundError(
        f"Image not found:\n{os.path.abspath(image_path)}"
    )


# ============================================================
# 5. LOAD IMAGE
# ============================================================

original_image = Image.open(
    image_path
).convert("RGB")

print("\nImage Loaded Successfully!")
print("Image Size:", original_image.size)


# ============================================================
# 6. IMAGE TRANSFORMATION
#
# IMPORTANT:
# This MUST MATCH YOUR TRAINING DATASET
#
# Your current dataset uses:
# Resize -> ToTensor
#
# DO NOT ADD NORMALIZATION HERE
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================================================
# 7. PREPARE INPUT TENSOR
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
# 8. CNN PREDICTION
# ============================================================

with torch.no_grad():

    outputs = model(
        image_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, predicted = torch.max(
        probabilities,
        dim=1
    )


# ============================================================
# 9. GET PREDICTED CLASS
# ============================================================

predicted_index = predicted.item()

predicted_class = class_names[
    predicted_index
]

confidence_value = (
    confidence.item() * 100
)


# ============================================================
# 10. PRINT CNN PREDICTION
# ============================================================

print("\n======================================")
print("          CNN PREDICTION")
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
# 11. PRINT ALL CLASS PROBABILITIES
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
# 12. PRINT RAW MODEL OUTPUTS
# ============================================================

print("\nRaw Model Outputs:")
print(outputs)


# ============================================================
# 13. PREDICTED CLASS INDEX
# ============================================================

print(
    "\nPredicted Class Index:",
    predicted_index
)

print(
    "Predicted Class:",
    predicted_class
)


# ============================================================
# 14. SELECT LAST CONVOLUTIONAL LAYER
#
# CNN ARCHITECTURE:
#
# features[0]  -> Conv2d 3 → 16
# features[1]  -> BatchNorm
# features[2]  -> ReLU
# features[3]  -> MaxPool
#
# features[4]  -> Conv2d 16 → 32
# features[5]  -> BatchNorm
# features[6]  -> ReLU
# features[7]  -> MaxPool
#
# features[8]  -> Conv2d 32 → 64
# features[9]  -> BatchNorm
# features[10] -> ReLU
# features[11] -> MaxPool
#
# features[12] -> Conv2d 64 → 128  ← TARGET
# features[13] -> BatchNorm
# features[14] -> ReLU
# features[15] -> MaxPool
#
# ============================================================

target_layers = [
    model.features[12]
]

print("\nTarget Layer:")
print(target_layers[0])


# ============================================================
# 15. CREATE GRAD-CAM
# ============================================================

cam = GradCAM(
    model=model,
    target_layers=target_layers
)


# ============================================================
# 16. TARGET THE PREDICTED CLASS
#
# Grad-CAM will explain:
#
# "Why did the CNN predict THIS class?"
#
# ============================================================

targets = [
    ClassifierOutputTarget(
        predicted_index
    )
]


# ============================================================
# 17. GENERATE GRAD-CAM
# ============================================================

print("\nGenerating Grad-CAM...")

grayscale_cam = cam(
    input_tensor=image_tensor,
    targets=targets
)

grayscale_cam = grayscale_cam[0]

print("Grad-CAM Generated Successfully!")


# ============================================================
# 18. PREPARE IMAGE FOR VISUALIZATION
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
# 19. CREATE GRAD-CAM OVERLAY
# ============================================================

visualization = show_cam_on_image(
    display_image,
    grayscale_cam,
    use_rgb=True
)


# ============================================================
# 20. CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# 21. SAVE GRAD-CAM RESULT
# ============================================================

gradcam_path = (
    "results/gradcam_result.jpg"
)

Image.fromarray(
    visualization
).save(
    gradcam_path
)


# ============================================================
# 22. SAVE HEATMAP SEPARATELY
# ============================================================

heatmap_path = (
    "results/gradcam_heatmap.jpg"
)

heatmap_image = (
    grayscale_cam * 255
).astype(
    np.uint8
)

Image.fromarray(
    heatmap_image
).save(
    heatmap_path
)


# ============================================================
# 23. DISPLAY THREE VISUALIZATIONS
# ============================================================

plt.figure(
    figsize=(15, 5)
)


# ---------------- Original MRI ----------------

plt.subplot(
    1, 3, 1
)

plt.imshow(
    display_image
)

plt.title(
    "Original MRI"
)

plt.axis("off")


# ---------------- Grad-CAM Heatmap ----------------

plt.subplot(
    1, 3, 2
)

plt.imshow(
    grayscale_cam,
    cmap="jet"
)

plt.title(
    "Grad-CAM Heatmap"
)

plt.axis("off")


# ---------------- Overlay ----------------

plt.subplot(
    1, 3, 3
)

plt.imshow(
    visualization
)

plt.title(
    f"Grad-CAM Overlay\n"
    f"{predicted_class} "
    f"({confidence_value:.2f}%)"
)

plt.axis("off")


plt.tight_layout()


# ============================================================
# 24. SAVE COMPLETE FIGURE
# ============================================================

figure_path = (
    "results/gradcam_analysis.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 25. SHOW FIGURE
# ============================================================

plt.show()


# ============================================================
# 26. FINAL OUTPUT
# ============================================================

print("\n======================================")
print("     GRAD-CAM COMPLETED SUCCESSFULLY")
print("======================================")

print(
    "\nPrediction:",
    predicted_class
)

print(
    f"Confidence: {confidence_value:.2f}%"
)

print(
    "\nGrad-CAM Overlay saved to:"
)

print(
    os.path.abspath(
        gradcam_path
    )
)

print(
    "\nHeatmap saved to:"
)

print(
    os.path.abspath(
        heatmap_path
    )
)

print(
    "\nComplete analysis saved to:"
)

print(
    os.path.abspath(
        figure_path
    )
)

print("\nProcess finished successfully.")