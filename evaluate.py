import os

import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from models.cnn_model import BrainTumorCNN
from dataset.dataloader import test_loader


# =====================================================
# Device
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)


# =====================================================
# Model
# =====================================================

model = BrainTumorCNN().to(device)

model.load_state_dict(
    torch.load(
        "brain_tumor_cnn_v2.pth",
        map_location=device
    )
)

model.eval()


# =====================================================
# Evaluation
# =====================================================

correct = 0
total = 0

all_labels = []
all_predictions = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )


accuracy = (
    100 * correct / total
)


# =====================================================
# Create Results Directory
# =====================================================

os.makedirs(
    "results",
    exist_ok=True
)


# =====================================================
# Accuracy
# =====================================================

with open(
    "results/accuracy.txt",
    "w"
) as f:

    f.write(
        f"Test Accuracy: "
        f"{accuracy:.2f}%"
    )


print(
    f"\nTest Accuracy: "
    f"{accuracy:.2f}%"
)


# =====================================================
# Classification Report
# =====================================================

class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names
)


print("\nClassification Report:\n")

print(report)


with open(
    "results/classification_report.txt",
    "w"
) as f:

    f.write(report)


# =====================================================
# Confusion Matrix
# =====================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)


fig, ax = plt.subplots(
    figsize=(7, 6)
)


disp.plot(ax=ax)


plt.title(
    "Brain Tumor Confusion Matrix"
)


plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)


plt.show()


print(
    "\nEvaluation completed."
)