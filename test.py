import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import csv
import pandas as pd
import numpy as np
def emo_type(t):
    if t==0:
        type = 'angry'
    elif t==1:
        type = 'disgust'
    elif t==2:
        type = 'fear'
    elif t==3:
        type = 'happy'
    elif t==4:
        type = 'neutral'
    elif t==5:
        type = 'sad'
    elif t==6:
        type = 'surprise'
    return type

batch_size = 16

net = models.resnet18(pretrained=True)
test_directory='emotion/test'
Image_path = 'E:\desk/face/emotion/test'
dataset_path = 'emotion/test'
model_path='models_resnet18_ep300/emotion_model_40_a0.875.pt'
net = torch.load(model_path)


image_transforms = {
'test': transforms.Compose([
        transforms.RandomResizedCrop(size=48, scale=(0.8, 1.0)),
        transforms.CenterCrop(size=45),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
data={
'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}
test_data_size = len(data['test'])
test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=False)

for folder,subfolder,filenames in os.walk(dataset_path):
    print(filenames)


df = pd.read_csv('submission0.csv', encoding='utf-8')
k=0
for i,(inputs,labels) in enumerate(test_data):
    outputs=net(inputs)
    ret,predictions = torch.max(outputs.data, 1)
    print(predictions)
    for j in range(len(predictions)):
        sample_fname, _ = test_data.dataset.samples[k]
        print(sample_fname)
        pre=int(predictions[j])
        emotion = emo_type(pre)
        file_name = filenames[k]
        print(file_name,emotion)
        print(k)
        k+=1
        for m in range(len(filenames)):
            if df['file_name'].loc[m] == file_name:
                df['class'].loc[m] = emotion
                break



df.to_csv('submission.csv', encoding='utf-8',index=False)
