from torch.utils.data import DataLoader

from dataset.tumor_dataset import (
    train_dataset,
    test_dataset
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))

print("Training batches:", len(train_loader))
print("Testing batches:", len(test_loader))