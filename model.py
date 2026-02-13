import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

## layer defining

data_dir = "dataset"  
batch_size = 16
num_epochs = 8
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# layer & batches

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}


def main():
    # loading data
    for phase in ["train", "val", "test"]:
        path = os.path.join(data_dir, phase)
        if not os.path.exists(path) or not any(os.path.isdir(os.path.join(path, d)) for d in os.listdir(path)):
            raise FileNotFoundError(f"Missing or empty dataset directory: {path}")


    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x])
        for x in ["train", "val", "test"]
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2)  
        for x in ["train", "val", "test"]
    }
    class_names = image_datasets["train"].classes
    num_classes = len(class_names)

    print("Classes:", class_names)

    
    #model setup - mobilenet_v2 
    
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False  # freeze backbone

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    #loss and optimizing
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    #training and validating

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    #saving model

    torch.save(model.state_dict(), "butterfly_classifier.pth")
    print("\nTraining complete. Model saved as butterfly_classifier.pth")

    #final testing

    model.eval()
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / len(image_datasets["test"])
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    

#main frame part -

if __name__ == "__main__":
    main()