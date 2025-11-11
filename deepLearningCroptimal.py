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
# NUM_EPOCHS = 50
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
# NUM_EPOCHS = 50
# LEARNING_RATE = 1e-4
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if DEVICE.type == 'cuda':
#     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
# else:
#     print("Using CPU")

# # ==== TRANSFORMS ====
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

# # ==== MODEL: MobileNetV3-Large ====
# model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
# # Replace the classifier for binary classification
# num_ftrs = model.classifier[3].in_features
# model.classifier[3] = nn.Linear(num_ftrs, NUM_CLASSES)
# model = model.to(DEVICE)

# # ==== LOSS AND OPTIMIZER ====
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# # ==== AMP SCALER ====
# scaler = torch.amp.GradScaler()

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
# torch.save(best_model.state_dict(), "mobilenetv3_two_class_amp.pth")
# print("Model saved to mobilenetv3_two_class_amp.pth")




# # train_inception_two_classes_amp_log.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, models, transforms
# from torch.utils.data import DataLoader
# import time
# import copy
# import os
# import csv

# # ==== CONFIGURATION ====
# DATA_DIR = "Croptimal/resnet"          # Root folder containing 'train' and 'val' subfolders
# BATCH_SIZE = 16
# NUM_CLASSES = 2
# NUM_EPOCHS = 50
# LEARNING_RATE = 1e-4
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if DEVICE.type == 'cuda':
#     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
# else:
#     print("Using CPU")

# # ==== TRANSFORMS ====
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(20),
#         transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize((299, 299)),
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

# # ==== MODEL: Inception v3 ====
# model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
# model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, NUM_CLASSES)
# model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
# model = model.to(DEVICE)

# # ==== LOSS, OPTIMIZER, SCHEDULER ====
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# # ==== AMP SCALER ====
# scaler = torch.amp.GradScaler()

# # ==== EARLY STOPPING ====
# early_stop_patience = 5

# # ==== TRAINING FUNCTION ====
# def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
#     since = time.time()
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#     epochs_no_improve = 0

#     # Prepare CSV log
#     log_path = "training_log.csv"
#     with open(log_path, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["epoch", "phase", "loss", "accuracy", "lr"])

#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch+1}/{num_epochs}")
#         print("-" * 20)
#         lr = optimizer.param_groups[0]['lr']

#         epoch_stats = {}

#         for phase in ['train', 'val']:
#             model.train() if phase == 'train' else model.eval()
#             running_loss = 0.0
#             running_corrects = 0

#             for inputs, labels in dataloaders[phase]:
#                 inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
#                 optimizer.zero_grad()

#                 with torch.amp.autocast(device_type=DEVICE.type):
#                     if phase == 'train':
#                         outputs, aux_outputs = model(inputs)
#                         loss1 = criterion(outputs, labels)
#                         loss2 = criterion(aux_outputs, labels)
#                         loss = loss1 + 0.4 * loss2
#                     else:
#                         outputs = model(inputs)
#                         loss = criterion(outputs, labels)
#                     _, preds = torch.max(outputs, 1)

#                 if phase == 'train':
#                     scaler.scale(loss).backward()
#                     scaler.step(optimizer)
#                     scaler.update()

#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             epoch_stats[phase] = (epoch_loss, epoch_acc)

#             if phase == 'val':
#                 scheduler.step(epoch_loss)

#             print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

#             # Log each phase
#             with open(log_path, mode='a', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow([epoch + 1, phase, epoch_loss, epoch_acc.item(), lr])

#         # Early stopping check
#         val_acc = epoch_stats['val'][1]
#         if val_acc > best_acc:
#             best_acc = val_acc
#             best_model_wts = copy.deepcopy(model.state_dict())
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1

#         print(f"No improvement for {epochs_no_improve} epochs.\n")

#         if epochs_no_improve >= early_stop_patience:
#             print("Early stopping triggered — validation accuracy not improving.")
#             break

#     time_elapsed = time.time() - since
#     print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
#     print(f"Best val Acc: {best_acc:.4f}")
#     print(f"Training log saved to {log_path}")

#     model.load_state_dict(best_model_wts)
#     return model

# # ==== RUN TRAINING ====
# best_model = train_model(model, dataloaders, criterion, optimizer, scheduler, NUM_EPOCHS)

# # ==== SAVE MODEL ====
# torch.save(best_model.state_dict(), "inceptionv3_two_class_amp_log.pth")
# print("Model saved to inceptionv3_two_class_amp_log.pth")




# train_swint_two_classes_amp_log.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import copy
import os
import csv

# ==== CONFIGURATION ====
DATA_DIR = "Croptimal/resnet"          # Root folder containing 'train' and 'val' subfolders
BATCH_SIZE = 16
NUM_CLASSES = 2
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# ==== TRANSFORMS ====
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
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

# ==== MODEL: Swin Transformer (Tiny) ====
model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
num_ftrs = model.head.in_features
model.head = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(DEVICE)

# ==== LOSS, OPTIMIZER, SCHEDULER ====
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ==== AMP SCALER ====
scaler = torch.amp.GradScaler()

# ==== EARLY STOPPING ====
early_stop_patience = 5

# ==== TRAINING FUNCTION ====
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    # Prepare CSV log
    log_path = "training_log.csv"
    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "phase", "loss", "accuracy", "lr"])

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 20)
        lr = optimizer.param_groups[0]['lr']
        epoch_stats = {}

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.amp.autocast(device_type=DEVICE.type):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_stats[phase] = (epoch_loss, epoch_acc)

            if phase == 'val':
                scheduler.step(epoch_loss)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Log each phase
            with open(log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, phase, epoch_loss, epoch_acc.item(), lr])

        # Early stopping and checkpointing
        val_acc = epoch_stats['val'][1]
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"No improvement for {epochs_no_improve} epochs.\n")

        if epochs_no_improve >= early_stop_patience:
            print("⏹️ Early stopping triggered — validation accuracy not improving.")
            break

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")
    print(f"Training log saved to {log_path}")

    model.load_state_dict(best_model_wts)
    return model

# ==== RUN TRAINING ====
best_model = train_model(model, dataloaders, criterion, optimizer, scheduler, NUM_EPOCHS)

# ==== SAVE FINAL MODEL ====
torch.save(best_model.state_dict(), "swint_two_class_amp_log.pth")
print("Model saved to swint_two_class_amp_log.pth")
