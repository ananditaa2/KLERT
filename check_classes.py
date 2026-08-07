from dataset.dataloader import train_loader, test_loader


print("======================================")
print("       DATASET CLASS CHECK")
print("======================================")


print("\nTraining classes:")
print(train_loader.dataset.classes)


print("\nTraining class_to_idx:")
print(train_loader.dataset.class_to_idx)


print("\nTesting classes:")
print(test_loader.dataset.classes)


print("\nTesting class_to_idx:")
print(test_loader.dataset.class_to_idx)


print("\n======================================")
print("              DONE")
print("======================================")