from torch import nn
from torchvision import models

def ImageClassifier():
    model = models.resnet18(pretrained=True)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(*[
        nn.Linear(in_features=num_features, out_features=2),
        nn.LogSoftmax(dim=1)
    ])

    for param in model.parameters():
        param.requires_grad = True
    
    return model