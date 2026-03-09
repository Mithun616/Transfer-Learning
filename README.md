# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset

Transfer learning uses a pre-trained model to solve a new problem with limited data. In this experiment, a pre-trained VGG-19 model is used to classify images from the given dataset (chip_data.zip). The dataset contains images organized into different classes and is divided into training and testing folders. Images are resized to 224 × 224 to match the input size required by the VGG-19 model.

## DESIGN STEPS
### STEP 1:

Load the dataset using ImageFolder, apply transformations such as resizing and converting images to tensors, and create DataLoaders for batch processing.

### STEP 2:

Load the pre-trained VGG-19 model, freeze its feature extraction layers, and modify the final fully connected layer to match the number of classes in the dataset.

### STEP 3:

Train the model using the training dataset and evaluate its performance on the test dataset using metrics such as training loss, confusion matrix, and classification report.

## PROGRAM
Include your code here
```python

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

!unzip -oq ./chip_data.zip -d data
dataset_path = "./data/dataset/"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)

def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0)
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.show()

show_sample_images(train_dataset)

print(f"Total number of training samples: {len(train_dataset)}")
first_image, label = train_dataset[0]
print(f"Shape of the first image: {first_image.shape}")

print(f"Total number of testing samples: {len(test_dataset)}")
first_image_test, label_test = test_dataset[0]
print(f"Shape of the first test image: {first_image_test.shape}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
from torchvision.models import vgg19, VGG19_Weights
model = vgg19(weights=VGG19_Weights.DEFAULT)
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
from torchsummary import summary
summary(model, input_size=(3, 224, 224))
for param in model.features.parameters():
    param.requires_grad = False
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    print("Name: Mithun Kumar G")
    print("Register Number: 212224230160")

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses)
    plt.plot(range(1, num_epochs + 1), val_losses)
    plt.show()
train_model(model, train_loader, test_loader, num_epochs=10)
def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)

            prob = torch.sigmoid(outputs)
            predicted = (prob > 0.5).int().squeeze()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    print("Name: Mithun Kumar G")
    print("Register Number: 212224230160")

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.show()

    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
test_model(model, test_loader)
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        prob = torch.sigmoid(output)
        predicted = (prob > 0.5).int().item()

    class_names = dataset.classes
    image_to_display = transforms.ToPILImage()(image)
    plt.imshow(image_to_display)
    plt.title(f'Actual: {class_names[label]} | Predicted: {class_names[predicted]}')
    plt.axis("off")
    plt.show()

    print("Name: Mithun Kumar G")
    print("Register Number: 212224230160")
predict_image(model, image_index=55, dataset=test_dataset)
predict_image(model, image_index=25, dataset=test_dataset)

```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot

<img width="736" height="730" alt="image" src="https://github.com/user-attachments/assets/dc670e5c-9914-40b3-8fc3-9e9a93d2bfcf" />

### Confusion Matrix

<img width="556" height="462" alt="image" src="https://github.com/user-attachments/assets/2b905bd8-3682-4ba5-ae1c-95c2f71a1b14" />

### Classification Report

<img width="520" height="163" alt="image" src="https://github.com/user-attachments/assets/1140d1bb-ca53-49ef-9529-c854d6e45693" />

### New Sample Prediction

<img width="435" height="451" alt="image" src="https://github.com/user-attachments/assets/a48d5d98-a5ca-40df-b043-e2484491f108" />

<img width="404" height="451" alt="image" src="https://github.com/user-attachments/assets/912f7eb9-2027-46d9-9913-c4d6fd08fbb1" />

## RESULT

Thus, transfer learning was successfully implemented using the pre-trained VGG-19 architecture. The model was trained on the given dataset and its performance was evaluated using metrics such as training loss, confusion matrix, and classification report, demonstrating its ability to classify the images correctly.
