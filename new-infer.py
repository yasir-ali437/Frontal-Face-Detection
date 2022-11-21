import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import time
import os
import numpy as np
import cv2
import bleedfacedetector as fd
import os
from PIL import Image


# img = cv2.imread('./inference1.jpg')

# img = cv2.resize(img, (64, 64))

# cv2.imshow( "FRAME", img)

# cv2.waitKey(0)

# cv2.imwrite("1.png", img)

def Model(model):

    n_inputs = model.fc.in_features

    classifier=nn.Linear(n_inputs, 2) 

    model.fc = classifier

    return model


def inference():
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    mapper = ['Frontal', 'Side']
    path = os.getcwd() + '/image-inference'
    pathDir = os.listdir(path)

    correct = 0

    total = len(pathDir)

    for image_ in pathDir:
        frame = cv2.imread(path + '/' + image_ )
        # print('THIS', image_, frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # faces = fd.ssd_detect(frame)

        # for face in faces:

        #     x=face[0]
        #     y=face[1]
        #     w=face[2]
        #     h=face[3]

        #     x2=x+w
        #     y2=y+h

            # cropped_face = frame[y:y2, x:x2]

        img_crop = Image.fromarray(frame)
        transform_face = transform(img_crop)
        img = torch.unsqueeze(transform_face, 0)

        t1=time.time()

        out = model(img.cuda())

        # print("inference time is: ", time.time()-t1)

        _, index = torch.max(out, 1)
        label = index.item()
        output = mapper[label]

        if output == 'Side':
            correct+=1

        # c1, c2 = (int(x),int(y)),(int(x2), int(y2))

        # cv2.rectangle(frame, c1, c2, (0, 0, 255), 2)
        # v2.putText(frame, output, (c1[0], c1[1] - 2), 0, 2 / 3, (0,255,0), 2, lineType=cv2.LINE_AA)
        # cv2.imshow("frame", frame)

    print("acc", (correct/total) * 100)



   
          

if __name__ == '__main__':


    model = models.resnet18(pretrained=True) 
    model= Model(model)

    try:
        model.load_state_dict(torch.load('./ResNet18_best.pth'))
        model.cuda()
        model.eval()

    except Exception as e:
        print('ERROR: model not loaded properly')

    inference()
