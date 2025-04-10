import os
import copy
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import models, transforms
from sklearn.model_selection import KFold
from PIL import Image

# -----------------------------------------------
# CONFIGURATION
# -----------------------------------------------
# IMG_DIR = 'combined_imgs'
IMG_DIR = '2025S_imgs'
IMG_SIZE = 224
LABELS_FILE = os.path.join(IMG_DIR, 'labels.txt')
NUM_SIGNS = 11
NUM_CLASSES = 6
BATCH_SIZE = 32
NUM_EPOCHS = 2
LEARNING_RATE = 1e-4
MODEL_SAVE_DIR = 'model_parameters'


# -----------------------------------------------
# CUSTOM DATASET
# -----------------------------------------------

# Labels
def load_labels(filepath):
    labels = {}
    with open(filepath, 'r') as f:
        for line in f:
            img_id, label = line.strip().split(',')
            img_id = f"{int(img_id.strip()):03}"  # Ensure 3-digit zero-padded
            labels[img_id] = int(label.strip())
    return labels

# Load labels
labels_dict = load_labels(LABELS_FILE)

# Dataset
class WallsignDataset(Dataset):
    def __init__(self, img_dir, labels_dict, transform=None):
        self.img_dir = img_dir
        self.labels_dict = labels_dict
        self.image_ids = list(labels_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert("RGB")
        label = self.labels_dict[img_id]
        if self.transform:
            image = self.transform(image)
        return image, label

# Load dataset
dataset = WallsignDataset(IMG_DIR, labels_dict)


# -----------------------------------------------
# TRANSFORMERS & LOADERS
# -----------------------------------------------

# Values for normalization created seperately
mean = [0.66400695, 0.45201, 0.4441439]
std = [0.13950367, 0.15291268, 0.14623028]

# train with data augmentation
train_transformer = transforms.Compose([ # WORSE PERFORMANCE # 
    # NO PERFORMANCE IMPROVE # transforms.RandomPerspective(distortion_scale=0.1, p=0.5, fill=0),  # Add perspective shift
    # NO PERFORMANCE IMPROVE # transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.9, 1.0)), # RandomCrop
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust color properties
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
    transforms.RandomRotation(degrees=10, fill=0),  # Rotates randomly between + and - degree and fills new pixels with black
    transforms.Resize(IMG_SIZE), # Resize to models needs
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean, std) # we have to calculate these values for our dataset
])
# test without data augmentation
test_transformer = transforms.Compose([   
    transforms.Resize(IMG_SIZE), # Resize to models needs
    transforms.CenterCrop(IMG_SIZE), # shouldn't do anything
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Apply transformations to each dataset
train_dataset.dataset.transform = train_transformer
test_dataset.dataset.transform = test_transformer

# Create DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# -----------------------------------------------
# LOAD AND MODIFY MODEL
# -----------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
# model.fc = nn.Sequential(
#     nn.Linear(model.fc.in_features, NUM_SIGNS),
#     nn.ReLU(),
#     nn.Linear(NUM_SIGNS, NUM_CLASSES)
# )
model = model.to(device)
# model.load_state_dict(torch.load('model_parameters/resnet18_20250408_162734_acc_1.000.pth'))


# -----------------------------------------------
# TRAINING LOOP
# -----------------------------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Function for training a single epoch
def train_loop(model, train_loader, loss_fn, optimizer, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0

    # Iterate over the training dataset
    for inputs, labels in train_loader:
    # for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss and accuracy
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)

    # Calculate epoch loss and accuracy
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    return epoch_loss, epoch_acc


# Function for evaluating a single epoch
def test_loop(model, test_loader, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    running_corrects = 0

    # Iterate over the validation dataset
    for inputs, labels in train_loader:
    # for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)

        # Accumulate loss and accuracy
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)

    # Calculate epoch loss and accuracy
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.double() / len(test_loader.dataset)

    return epoch_loss, epoch_acc

# Main function for training the model
def train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, num_epochs, device):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train phase
        train_loss, train_acc = train_loop(model, train_loader, loss_fn, optimizer, device)
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')

        # Test phase
        val_loss, val_acc = test_loop(model, test_loader, loss_fn, device)
        print(f'Test Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # Check if the current model has the best accuracy on validation
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        # Step the learning rate scheduler
        scheduler.step()

    print(f"\nBest Validation Accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model, best_acc

def train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, num_epochs):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}:")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = test_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.5f} Acc: {epoch_acc:.5f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\nBest Validation Accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model, best_acc

# -----------------------------------------------
# RUN TRAINING
# -----------------------------------------------
trained_model, best_val_acc = train_model(
    model, train_loader, test_loader,
    loss_fn, optimizer, scheduler,
    num_epochs=NUM_EPOCHS
)

# -----------------------------------------------
# SAVE TRAINED MODEL
# -----------------------------------------------
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f"resnet18_{timestamp}_acc_{best_val_acc:.3f}.pth"
save_path = os.path.join(MODEL_SAVE_DIR, model_filename)
torch.save(trained_model.state_dict(), save_path)
print(f"\n Model saved to: {save_path}")
