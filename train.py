import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


import time
import os
import numpy as np
import copy

# import matplotlib.pyplot as plt


data_dir = 'Face_Position_dataset'

train_transforms = transforms.Compose([transforms.Resize((64, 64)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize((64, 64)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/val', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)


dataloaders_dict = {'train': trainloader, 'val':testloader}
for keys,items in dataloaders_dict.items():
    print(keys)
    
print(dataloaders_dict['train'].dataset.class_to_idx)    

device = "cpu"

model = models.resnet18(pretrained=True) 


def freeze(model):

    for param in model.parameters():

        param.require_grad = False
        
    return model

def transfer_learning(model):
    
    n_inputs = model.fc.in_features
    
    model.fc=nn.Linear(n_inputs, 2) # Front and Side classes
    
    
    return model


model=freeze(model)
model=transfer_learning(model)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(), lr=0.01)
model.to(device)


num_epochs=40

since = time.time()

train_acc_history = []
train_loss_history = []
val_acc_history = []
val_loss_history = []

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
global inputs_, labels_

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0


        for inputs, labels in dataloaders_dict[phase]:
            inputs = inputs.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):

                output = model(inputs)

                loss = criterion(output, labels)
                _, preds = torch.max(output, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders_dict[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "ResNet18_best.pth")
            
        if phase == 'train':
            train_acc_history.append(epoch_acc)
            train_loss_history.append(epoch_loss)  

        if phase == 'val':
            val_acc_history.append(epoch_acc)
            val_loss_history.append(epoch_loss)


time_elapsed = time.time() - since

print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))