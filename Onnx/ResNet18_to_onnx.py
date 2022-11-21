import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18(pretrained=True) 

def Model(model):

    n_inputs = model.fc.in_features

    classifier=nn.Linear(n_inputs, 2) 

    model.fc = classifier

    return model

model= Model(model)

model.load_state_dict(torch.load('./pretrained-weights/ResNet18_best.pth'))

def To_Onnx():
    
    dummy_input = torch.randn(32, 3, 64, 64)

    torch.onnx.export(model, dummy_input, "frontal_face_classifier_ResNet18.onnx", verbose=True)


          

if __name__ == '__main__':
    To_Onnx()
