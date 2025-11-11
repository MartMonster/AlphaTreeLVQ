# # train_resnet50_two_classes.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, models, transforms
# from torch.utils.data import DataLoader
# import time
# import copy
# import os

# # ==== CONFIGURATION ====
# DATA_DIR = "Croptimal/resnet"          # Root folder containing 'train' and 'val' subfolders
# BATCH_SIZE = 16
# NUM_CLASSES = 2
# NUM_EPOCHS = 100
# LEARNING_RATE = 1e-4
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if DEVICE.type == 'cuda':
#     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
# else:
#     print("Using CPU")

# # ==== TRANSFORMS ====
# # Swin-T expects at least 224x224, so 256x256 is fine
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(15),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ]),
# }

# # ==== LOAD DATA ====
# image_datasets = {
#     x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
#     for x in ['train', 'val']
# }
# dataloaders = {
#     x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
#     for x in ['train', 'val']
# }
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes

# print(f"Classes: {class_names}")
# print(f"Dataset sizes: {dataset_sizes}")

# # ==== MODEL: Swin Transformer Tiny ====
# model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
# num_ftrs = model.head.in_features
# model.head = nn.Linear(num_ftrs, NUM_CLASSES)
# model = model.to(DEVICE)

# # model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# # # Replace the final layer for binary classification
# # num_ftrs = model.fc.in_features
# # model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
# # model = model.to(DEVICE)

# # ==== LOSS AND OPTIMIZER ====
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# # ==== AMP SCALER ====
# scaler = torch.amp.GradScaler(device=DEVICE.type)

# # ==== TRAINING FUNCTION ====
# def train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS):
#     since = time.time()
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0

#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch+1}/{num_epochs}")
#         print("-" * 20)

#         for phase in ['train', 'val']:
#             model.train() if phase == 'train' else model.eval()
#             running_loss = 0.0
#             running_corrects = 0

#             for inputs, labels in dataloaders[phase]:
#                 inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
#                 optimizer.zero_grad()

#                 # Use autocast for mixed precision
#                 with torch.amp.autocast(device_type=DEVICE.type):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                 if phase == 'train':
#                     # Scale the loss for mixed precision
#                     scaler.scale(loss).backward()
#                     scaler.step(optimizer)
#                     scaler.update()

#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())

#         print()

#     time_elapsed = time.time() - since
#     print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
#     print(f"Best val Acc: {best_acc:.4f}")

#     model.load_state_dict(best_model_wts)
#     return model

# # ==== RUN TRAINING ====
# best_model = train_model(model, dataloaders, criterion, optimizer, NUM_EPOCHS)

# # ==== SAVE MODEL ====
# torch.save(best_model.state_dict(), "swinT_two_class.pth")
# print("Model saved to swinT_two_class.pth")


# train_mobilenet_two_classes_amp.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import copy
import os

# ==== CONFIGURATION ====
DATA_DIR = "Croptimal/resnet"          # Root folder containing 'train' and 'val' subfolders
BATCH_SIZE = 16
NUM_CLASSES = 2
NUM_EPOCHS = 100
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# ==== TRANSFORMS ====
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# ==== LOAD DATA ====
image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
    for x in ['train', 'val']
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    for x in ['train', 'val']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(f"Classes: {class_names}")
print(f"Dataset sizes: {dataset_sizes}")

# ==== MODEL: MobileNetV3-Large ====
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
# Replace the classifier for binary classification
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(DEVICE)

# ==== LOSS AND OPTIMIZER ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# ==== AMP SCALER ====
scaler = torch.amp.GradScaler()

# ==== TRAINING FUNCTION ====
def train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                # Use autocast for mixed precision
                with torch.amp.autocast(device_type=DEVICE.type):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model

# ==== RUN TRAINING ====
best_model = train_model(model, dataloaders, criterion, optimizer, NUM_EPOCHS)

# ==== SAVE MODEL ====
torch.save(best_model.state_dict(), "mobilenetv3_two_class_amp.pth")
print("Model saved to mobilenetv3_two_class_amp.pth")
