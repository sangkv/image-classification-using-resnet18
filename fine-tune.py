import os
import copy
from tqdm import tqdm
from PIL import Image

import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from models import ImageClassifier

# Check GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Data
# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset & DataLoader
# Dataset Class
class CatDogDataset(Dataset):
    def __init__(self, dir, transform = None):
        self.dir = dir
        self.transform = transform
        self.list_dir_images = os.listdir(dir)
    
    def __len__(self):
        return len(self.list_dir_images)
    
    def __getitem__(self, index):
        name = self.list_dir_images[index].split('.')[0]
        label = 0 if name == 'cat' else 1

        image_path = os.path.join(self.dir, self.list_dir_images[index])
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

# Prepare data and dataloader
TRAIN_PATH = './data/train/'
VAL_PATH = './data/val/'
train_data = CatDogDataset(TRAIN_PATH, transform)
val_data = CatDogDataset(VAL_PATH, transform)

NUM_BATCH = 32
train_dl = DataLoader(train_data, batch_size=NUM_BATCH, shuffle=True)
val_dl = DataLoader(val_data, batch_size=NUM_BATCH)

# 2. Model
model = ImageClassifier()
model.load_state_dict(torch.load('./weights/ResNet18_CatDog.pth'))
model.to(DEVICE)

# 3. Fine-tune
# Functions
# Validation Function
def validate(model, data):
    total = 0
    correct = 0

    for i, (images, labels) in tqdm(enumerate(data)):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        total += outputs.size(0)
        correct += torch.sum(preds == labels)
        
    return correct.item() * 100 / total

# Train Function
def train(model, criterion, optimizer, epochs=10):
    model.train()

    max_accuracy = 0
    best_model = copy.deepcopy(model)

    for epoch in range(epochs):
        for i, (images, labels) in tqdm(enumerate(train_dl)):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        val_accuracy = validate(model, val_dl)
        if val_accuracy > max_accuracy:
            best_model = copy.deepcopy(model)
            max_accuracy = val_accuracy
            print('saving best model with val_accuracy: ', val_accuracy)
        print('Epoch: ', epoch+1, ', val_accuracy: ', val_accuracy, '%')
    
    # Save Best Model
    torch.save(best_model.state_dict(), './weights/ResNet18_CatDog_final.pth')

# Train the model
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
train(model=model, criterion=criterion, optimizer=optimizer, epochs=3)

