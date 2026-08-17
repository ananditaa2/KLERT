import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ============================================================
# 1. Device and Dataset Setup
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Training Device: {device}")

data_dir = "data"
train_dir = os.path.join(data_dir, "Training")
test_dir = os.path.join(data_dir, "Testing")

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

print("[*] Loading datasets...")
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

class_names = train_dataset.classes
print(f"[*] Classes detected ({len(class_names)}): {class_names}")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)


# ============================================================
# 2. Training Function
# ============================================================
def train_and_evaluate(model, model_name, save_path, epochs=4, lr=1e-3):
    print(f"\n=======================================================")
    print(f"[*] Starting Transfer Learning for: {model_name}")
    print(f"=======================================================")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        start_t = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # Evaluation on Test Split
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_acc = (correct_val / total_val) * 100.0
        elapsed = time.time() - start_t

        print(f"Epoch [{epoch}/{epochs}] ({elapsed:.1f}s) - Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best checkpoint: {save_path} (Val Acc: {best_val_acc:.2f}%)")

    print(f"[*] Completed {model_name}. Final Best Accuracy: {best_val_acc:.2f}%\n")
    return best_val_acc


# ============================================================
# 3. Fine-Tune ResNet-18
# ============================================================
def setup_and_train_resnet():
    print("[*] Instantiating ResNet-18 with ImageNet Pretrained Backbone...")
    try:
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        resnet = models.resnet18(pretrained=True)

    # Unfreeze layer4 and classification head for optimal brain MRI transfer learning
    for param in resnet.parameters():
        param.requires_grad = False
    for param in resnet.layer4.parameters():
        param.requires_grad = True

    resnet.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(resnet.fc.in_features, len(class_names))
    )

    train_and_evaluate(resnet, "ResNet-18", "resnet18_brain_tumor.pth", epochs=4, lr=5e-4)


# ============================================================
# 4. Fine-Tune VGG-16
# ============================================================
def setup_and_train_vgg():
    print("[*] Instantiating VGG-16 with ImageNet Pretrained Backbone...")
    try:
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    except Exception:
        vgg = models.vgg16(pretrained=True)

    # Freeze base features, unfreeze final conv block and classifier
    for param in vgg.features[:24]:
        param.requires_grad = False
    for param in vgg.features[24:]:
        param.requires_grad = True

    vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, len(class_names))

    train_and_evaluate(vgg, "VGG-16", "vgg16_brain_tumor.pth", epochs=3, lr=1e-4)


if __name__ == "__main__":
    setup_and_train_resnet()
    setup_and_train_vgg()
    print("[✓] All pretrained models successfully fine-tuned and saved!")
