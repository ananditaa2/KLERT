import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Training Device: {device}", flush=True)

data_dir = "data"
train_dir = os.path.join(data_dir, "Training")
test_dir = os.path.join(data_dir, "Testing")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

print("[*] Loading datasets...", flush=True)
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

class_names = train_dataset.classes
print(f"[*] Classes ({len(class_names)}): {class_names}", flush=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

print("[*] Instantiating VGG-16 with pretrained weights...", flush=True)
vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
vgg.eval()

# Extract features
def extract_features(loader, name):
    print(f"[*] Pre-computing feature maps for {name} ({len(loader.dataset)} samples)...", flush=True)
    all_feats = []
    all_labels = []
    t0 = time.time()
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(loader):
            imgs = imgs.to(device)
            # Forward pass through features + avgpool
            feats = vgg.features(imgs)
            feats = vgg.avgpool(feats)
            feats = torch.flatten(feats, 1)
            all_feats.append(feats.cpu())
            all_labels.append(lbls)
            if (i + 1) % 20 == 0 or (i + 1) == len(loader):
                print(f"    Processed {min((i+1)*64, len(loader.dataset))}/{len(loader.dataset)} scans ({time.time()-t0:.1f}s)", flush=True)

    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)

train_x, train_y = extract_features(train_loader, "Training Set")
test_x, test_y = extract_features(test_loader, "Testing Set")

print(f"[*] Feature Extraction Complete! Train shape: {train_x.shape}, Test shape: {test_x.shape}", flush=True)

# Train Classifier
feat_train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
feat_test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=64, shuffle=False)

classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(4096, len(class_names))
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=1e-4)

print("\n[*] Training VGG-16 Classifier Head on Extracted Features...", flush=True)
epochs = 8
best_acc = 0.0

for epoch in range(1, epochs + 1):
    classifier.train()
    running_loss, correct, total = 0.0, 0, 0
    for x_b, y_b in feat_train_loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out = classifier(x_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x_b.size(0)
        _, preds = torch.max(out, 1)
        correct += (preds == y_b).sum().item()
        total += y_b.size(0)

    train_acc = (correct / total) * 100.0

    # Validation
    classifier.eval()
    c_val, t_val = 0, 0
    with torch.no_grad():
        for x_b, y_b in feat_test_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            out = classifier(x_b)
            _, preds = torch.max(out, 1)
            c_val += (preds == y_b).sum().item()
            t_val += y_b.size(0)

    val_acc = (c_val / t_val) * 100.0
    print(f"Epoch [{epoch}/{epochs}] - Train Loss: {running_loss/total:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%", flush=True)

    if val_acc > best_acc:
        best_acc = val_acc
        # Assemble full VGG-16 and save
        vgg.classifier = classifier
        torch.save(vgg.state_dict(), "vgg16_brain_tumor.pth")
        print(f"  --> Saved new best checkpoint: vgg16_brain_tumor.pth (Val Acc: {best_acc:.2f}%)", flush=True)

print(f"\n[✓] VGG-16 Fine-Tuning Successfully Completed! Final Val Accuracy: {best_acc:.2f}%", flush=True)
