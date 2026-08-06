from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    root="data/Training",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root="data/Testing",
    transform=transform
)

print("Training Classes:")
print(train_dataset.classes)

print("\nTraining Class to Index:")
print(train_dataset.class_to_idx)

print("\nTesting Classes:")
print(test_dataset.classes)

print("\nTesting Class to Index:")
print(test_dataset.class_to_idx)