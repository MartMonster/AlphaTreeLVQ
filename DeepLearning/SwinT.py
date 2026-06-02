import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import copy
import os
import csv
from tqdm import tqdm

# ==== CONFIGURATION ====
DATA_DIR = "../Croptimal/resnet"          # Root folder containing 'train' and 'val' subfolders
BATCH_SIZE = 16
NUM_CLASSES = 2
NUM_EPOCHS = 500
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

# ==== MODEL: Swin Tiny ====
MODELNAME = "swin_t"
model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
# Replace the final layer for binary classification
num_ftrs = model.head.in_features
model.head = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(DEVICE)

# ==== LOSS, OPTIMIZER, SCHEDULER ====
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ==== AMP SCALER ====
scaler = torch.amp.GradScaler()

# ==== TRAINING FUNCTION ====
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Prepare CSV log
    log_path = f"training_log_{MODELNAME}.csv"
    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "phase", "loss", "accuracy", "false_positive_rate", "false_negative_rate", "lr"])

    for epoch in tqdm(range(num_epochs)):
        # print(f"Epoch {epoch+1}/{num_epochs}")
        # print("-" * 20)
        lr = optimizer.param_groups[0]['lr']
        epoch_stats = {}

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0
            tp = fp = tn = fn = 0

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
                # Confusion matrix components (binary classification)
                tp += torch.sum((preds == 1) & (labels == 1)).item()
                tn += torch.sum((preds == 0) & (labels == 0)).item()
                fp += torch.sum((preds == 1) & (labels == 0)).item()
                fn += torch.sum((preds == 0) & (labels == 1)).item()


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            # Rates (safe division)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            epoch_stats[phase] = (epoch_loss, epoch_acc)

            if phase == 'val':
                scheduler.step(epoch_loss)

            # print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} FPR: {fpr:.4f} FNR: {fnr:.4f}")

            # Log each phase
            with open(log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, phase, epoch_loss, epoch_acc.item(), fpr, fnr, lr])

        val_acc = epoch_stats['val'][1]
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")
    print(f"Training log saved to {log_path}")

    model.load_state_dict(best_model_wts)
    return model

# ==== RUN TRAINING ====
best_model = train_model(model, dataloaders, criterion, optimizer, scheduler, NUM_EPOCHS)

# ==== SAVE FINAL MODEL ====
torch.save(best_model.state_dict(), f"{MODELNAME}_two_class_amp_log.pth")
print(f"Model saved to {MODELNAME}_two_class_amp_log.pth")