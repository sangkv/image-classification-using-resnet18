import os
from PIL import Image

import torch
from torchvision import transforms

from models import ImageClassifier

# Check GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model
model = ImageClassifier()
model.load_state_dict(torch.load('./weights/ResNet18_CatDog_final.pth'))
model.to(DEVICE)

# Transforms data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predict
def predict(model, image_path):
    model.eval()
    img = Image.open(image_path)

    with torch.no_grad():
        image = transform(img)
        image = image.unsqueeze(0)
        image = image.to(DEVICE)

        output = model(image)
    
    return output

pred = predict(model=model, image_path='./data/3408.jpg')
if torch.is_tensor(pred):
    pred_idx = torch.argmax(pred).item()

    pred_label = "cat" if pred_idx == 0 else "dog"
    
    print('Predicted: ', pred_label)
