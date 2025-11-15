
# We import the necessary libraries
import os
import sys
from pathlib import Path
from zipfile import ZipFile
from PIL import Image
import torch
from torchvision.io import read_image
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

# Torch setup (GPU preferred, if not CPU)
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# We set a seed for fair reproducibility when tuning
SEED = 7
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
g = torch.Generator()
g.manual_seed(SEED)

# STEP 1: Extraction and processing of the images
ROOT = Path(__file__).resolve().parent

zip_path = ROOT / "h2-data.zip"
out_dir  = ROOT / "h2-data"

if not out_dir.exists():
    with ZipFile(zip_path) as zf:
        zf.extractall(ROOT)

folders = {"02124075": 0, "02504458": 1, "03792782": 2, "07753592": 3}

train_imgs = []
train_labs = []
test_imgs = []
test_labs = []

# (224, 224)
# (192, 192)
image_size = (160, 160)
train_transform = transforms.Compose([
    transforms.Resize(image_size),
    #transforms.RandomResizedCrop(image_size),
    transforms.RandomHorizontalFlip(p=0.5),  # data augmentation helps generalization
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

with open(out_dir / "train.txt", "r", encoding="utf-8") as f:
    for line in f:
        name = line.strip()
        img = Image.open(out_dir / f"n{name[1:9]}/{name}").convert("RGB")
        img = train_transform(img)
        train_imgs.append(img)
        train_labs.append(folders[name[1:9]])

with open(out_dir / "test.txt", "r", encoding="utf-8") as f:
    for line in f:
        name = line.strip()
        img = Image.open(out_dir /     f"n{name[1:9]}/{name}").convert("RGB")
        img = test_transform(img)
        test_imgs.append(img)
        test_labs.append(folders[name[1:9]])

X_train = torch.stack(train_imgs)
Y_train = torch.tensor(train_labs)
X_test = torch.stack(test_imgs)
Y_test = torch.tensor(test_labs)

train_ds = TensorDataset(X_train, Y_train)
test_ds = TensorDataset(X_test, Y_test)
val_split = 0.1
val_len = int(len(train_ds)*val_split)
train_len = len(train_ds) - val_len

# We create a validation set with the training set (could also be done with scikit)
train_ds, val_ds = random_split(train_ds, [train_len, val_len], generator=g)

def make_loader(ds, shuffle):
    # To avoid bugs in Windows
    workers = 4
    if os.name == "nt":
        workers = 0
    return DataLoader(
        ds,
        batch_size=64,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )

train_loader = make_loader(train_ds, True)
val_loader = make_loader(val_ds, True)
test_loader = make_loader(test_ds, False)

# STEP 2: CNN model
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        '''
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        '''

        #self.reduce = nn.AdaptiveAvgPool2d((4,4))
        self.reduce = nn.AdaptiveAvgPool2d((2,2))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        #x = self.block4(x)

        x = self.reduce(x)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.fc2(x)

        return x
    
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss() # Since it is a multi-class classification (only one class per image)
# Optimizer options: Adam, SGD, AdamW
#optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
'''
optimizer = optim.SGD(
    model.parameters(),
    lr=1e-2,
    momentum=.9,
    weight_decay=1e-4
)
'''
#  Scheduler options: StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# Best pairing options:
# Adam / AdamW --> StepLR / CosineAnnealingLR
# SGD --> OneCycleLR / MultiStepLR
# Unstable val loss --> ReduceLROnPlateau
# The scaler prevents gradients from vanishing
scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

# STEP 3: Model training and testing 
best_val_accuracy = 0.0

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            # If possible, we move the computation to the GPU
            inputs = inputs.to(device, non_blocking = True)
            targets = targets.to(device, non_blocking = True)
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = model(inputs)

            preds = outputs.argmax(dim=1)
            #correct += (preds == targets).sum().item()
            correct += (preds == targets).sum().item()
            total   += targets.size(0)

    return correct / total

num_epochs = 8
    
for epoch in range(1, num_epochs + 1):
    model.train()
    correct, total = 0, 0
    for inputs, targets in train_loader:
        # If possible, we move the computation to the GPU
        inputs = inputs.to(device, non_blocking = True)
        targets = targets.to(device, non_blocking = True)

        optimizer.zero_grad(set_to_none = True)

        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = outputs.argmax(dim=1)
        #correct += (preds == targets).sum().item()
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_accuracy = correct / total
    val_accuracy = evaluate(model, val_loader)
    #scheduler.step()

    print(f"Epoch {epoch}: Training accuracy = {epoch_accuracy}, Validation accuracy = {val_accuracy}")


# Improvements made upon the first version of the model (100% training and 81.5% validation accuracies achieved)
# Problems: too slow and overfits
# Solutions: less epochs, batch regularization, reduction of image size, remove the scheduler

# In the end, we obtain a test accuracy of
test_accuracy = evaluate(model, test_loader)
print(f"Test accuracy: {test_accuracy}")
