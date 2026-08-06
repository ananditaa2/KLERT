import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn_model import BrainTumorCNN
from dataset.dataloader import train_loader, test_loader


# =====================================================
# Device
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)


# =====================================================
# Dataset Information
# =====================================================

print(
    "Number of training batches:",
    len(train_loader)
)

print(
    "Number of testing batches:",
    len(test_loader)
)


# =====================================================
# Model
# =====================================================

model = BrainTumorCNN().to(device)


# =====================================================
# Loss Function
# =====================================================

criterion = nn.CrossEntropyLoss()


# =====================================================
# Optimizer
# =====================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# =====================================================
# Training Settings
# =====================================================

num_epochs = 30

best_accuracy = 0.0


print("\nStarting Training...")


# =====================================================
# Training Loop
# =====================================================

for epoch in range(num_epochs):

    print(
        f"\n========== Epoch "
        f"{epoch + 1}/{num_epochs} =========="
    )

    model.train()

    running_loss = 0.0


    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()


    epoch_loss = (
        running_loss /
        len(train_loader)
    )

    print(
        f"Training Loss: "
        f"{epoch_loss:.4f}"
    )


    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    model.eval()

    correct = 0
    total = 0


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


    accuracy = (
        100 * correct / total
    )


    print(
        f"Validation Accuracy: "
        f"{accuracy:.2f}%"
    )


    # -------------------------------------------------
    # Save Best Model
    # -------------------------------------------------

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(
            model.state_dict(),
            "brain_tumor_cnn_v2.pth"
        )

        print(
            f"✅ New Best Model Saved! "
            f"({accuracy:.2f}%)"
        )


# =====================================================
# Training Complete
# =====================================================

print("\n==============================")

print("Training Finished!")

print(
    f"Best Validation Accuracy: "
    f"{best_accuracy:.2f}%"
)

print("==============================")