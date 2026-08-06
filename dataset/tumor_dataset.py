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


print("Classes:", train_dataset.classes)
print("Class to Index:", train_dataset.class_to_idx)