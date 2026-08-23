import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_fscore_support
)

from models.cnn_model import BrainTumorCNN
from dataset.dataloader import test_loader

# =====================================================
# Device Configuration
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Evaluation Device: {device}")

class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Create Results Directory
os.makedirs("results", exist_ok=True)

# =====================================================
# Model Factory
# =====================================================
def load_models():
    models_dict = {}

    # 1. Custom BrainTumorCNN
    cnn = BrainTumorCNN().to(device)
    cnn_path = "brain_tumor_cnn_v2.pth"
    if os.path.exists(cnn_path):
        cnn.load_state_dict(torch.load(cnn_path, map_location=device))
        print(f"[+] Loaded Custom CNN weights from {cnn_path}")
    cnn.eval()
    models_dict["CustomCNN"] = cnn

    # 2. ResNet-18
    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(resnet.fc.in_features, 4)
    )
    resnet = resnet.to(device)
    resnet_path = "resnet18_brain_tumor.pth"
    if os.path.exists(resnet_path):
        resnet.load_state_dict(torch.load(resnet_path, map_location=device))
        print(f"[+] Loaded ResNet18 weights from {resnet_path}")
    resnet.eval()
    models_dict["ResNet18"] = resnet

    # 3. VGG-16
    vgg = models.vgg16(weights=None)
    vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, 4)
    vgg = vgg.to(device)
    vgg_path = "vgg16_brain_tumor.pth"
    if os.path.exists(vgg_path):
        vgg.load_state_dict(torch.load(vgg_path, map_location=device))
        print(f"[+] Loaded VGG16 weights from {vgg_path}")
    vgg.eval()
    models_dict["VGG16"] = vgg

    return models_dict

# =====================================================
# Main Evaluation Loop
# =====================================================
def run_evaluation():
    models_dict = load_models()
    all_metrics = {}

    for model_name, model in models_dict.items():
        print(f"\n=================================================")
        print(f" Evaluating Model: {model_name}")
        print(f"=================================================")

        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = 100.0 * correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="macro"
        )

        all_metrics[model_name] = {
            "Accuracy": round(accuracy, 2),
            "F1": round(f1 * 100.0, 2),
            "Precision": round(precision * 100.0, 2),
            "Recall": round(recall * 100.0, 2)
        }

        print(f"Test Accuracy ({model_name}): {accuracy:.2f}%")
        print(f"Macro F1-Score: {f1 * 100.0:.2f}%")

        # Save Classification Report
        report = classification_report(
            all_labels, all_predictions, target_names=class_names
        )
        print("\nClassification Report:\n", report)

        report_path = f"results/classification_report_{model_name.lower()}.txt"
        with open(report_path, "w") as f:
            f.write(report)

        # Plot & Save Confusion Matrix
        cm = confusion_matrix(all_labels, all_predictions)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        plt.title(f"Brain Tumor Confusion Matrix - {model_name}")
        plt.tight_layout()
        cm_path = f"results/confusion_matrix_{model_name.lower()}.png"
        plt.savefig(cm_path, dpi=300)
        plt.close(fig)
        print(f"[+] Saved confusion matrix to {cm_path}")

    # =====================================================
    # Save Model Metrics to model_metrics.json
    # =====================================================
    metrics_json_path = "model_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[+] Updated {metrics_json_path} successfully.")

    # Save summary accuracy text file
    with open("results/accuracy.txt", "w") as f:
        for m_name, m_val in all_metrics.items():
            f.write(f"{m_name} Test Accuracy: {m_val['Accuracy']}%\n")

    # =====================================================
    # Plot Comparative Bar Chart
    # =====================================================
    model_keys = list(all_metrics.keys())
    accs = [all_metrics[k]["Accuracy"] for k in model_keys]
    f1s = [all_metrics[k]["F1"] for k in model_keys]
    precisions = [all_metrics[k]["Precision"] for k in model_keys]

    x = np.arange(len(model_keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, accs, width, label="Accuracy (%)", color="#38bdf8")
    ax.bar(x, f1s, width, label="Macro F1-Score (%)", color="#a855f7")
    ax.bar(x + width, precisions, width, label="Precision (%)", color="#22c55e")

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Multi-Model Evaluation Benchmark Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(model_keys)
    ax.set_ylim(80, 100)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(model_keys)):
        ax.text(x[i] - width, accs[i] + 0.3, f"{accs[i]}%", ha="center", fontsize=8)
        ax.text(x[i], f1s[i] + 0.3, f"{f1s[i]}%", ha="center", fontsize=8)
        ax.text(x[i] + width, precisions[i] + 0.3, f"{precisions[i]}%", ha="center", fontsize=8)

    plt.tight_layout()
    comparison_path = "results/model_comparison.png"
    plt.savefig(comparison_path, dpi=300)
    plt.close(fig)
    print(f"[+] Saved comparison plot to {comparison_path}")

    print("\n[✔] Evaluation completed successfully.")

if __name__ == "__main__":
    run_evaluation()