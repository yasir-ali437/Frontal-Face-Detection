import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes = 2)


model.load_state_dict(torch.load('pretrained-weights/EffNet_b0_best.pth'))

def To_Onnx():
    
    dummy_input = torch.randn(32, 3, 64, 64)

    torch.onnx.export(model, dummy_input, "frontal_face_classifier_EffNetb0.onnx", verbose=True)


          

if __name__ == '__main__':
    To_Onnx()
