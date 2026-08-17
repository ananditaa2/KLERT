import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Training Device: {device}")

data_dir = "data"
train_dir = os.path.join(data_dir, "Training")
test_dir = os.path.join(data_dir, "Testing")

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
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
print(f"[*] Classes ({len(class_names)}): {class_names}")

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

print("[*] Instantiating VGG-16 with pretrained weights...")
vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

# Freeze feature extractor for blazing fast CPU training
for param in vgg.features.parameters():
    param.requires_grad = False

# Replace classifier head for 4 classes
vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, len(class_names))
vgg = vgg.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg.classifier.parameters(), lr=1e-3)

epochs = 3
best_val_acc = 0.0
save_path = "vgg16_brain_tumor.pth"

print("\n[*] Starting Fast Classifier Fine-Tuning for VGG-16...")
for epoch in range(1, epochs + 1):
    t0 = time.time()
    vgg.train()
    running_loss, correct_train, total_train = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = vgg(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    train_acc = (correct_train / total_train) * 100.0

    # Validation
    vgg.eval()
    correct_val, total_val = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = vgg(images)
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    val_acc = (correct_val / total_val) * 100.0
    elapsed = time.time() - t0

    print(f"Epoch [{epoch}/{epochs}] ({elapsed:.1f}s) - Train Loss: {running_loss/total_train:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(vgg.state_dict(), save_path)
        print(f"  --> Saved new best checkpoint: {save_path} (Val Acc: {best_val_acc:.2f}%)")

print(f"\n[✓] VGG-16 Fine-Tuning Completed! Final Best Val Accuracy: {best_val_acc:.2f}%")
