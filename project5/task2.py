#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:26:37 2024

@author: taiwei
"""

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from script import Net
import matplotlib.pyplot as plt

model=Net()
model.load_state_dict(torch.load("mnist_model_wieghts.pth"))

for name,params in model.named_parameters():
    print(name," ",params.size())
    
fig=plt.figure(figsize=(10,10))
for i in range(10):
    ax=fig.add_subplot(5,4,2*i+1,xticks=[],yticks=[])
    #plt.imshow(model.conv1.weight[i,0].detach().numpy(),cmap='gray')
    plt.imshow(model.conv1.weight[i,0].detach().numpy())
    ax.set_title(f"filter {i+1}")


conv1=model.conv1.weight.detach().squeeze().numpy()
print(conv1.shape)
import numpy as np
np.save("filters",conv1)


import torchvision.transforms as transforms
import torchvision
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
image,label=trainset[0]
image=image.squeeze().numpy()*0.5+0.5
np.save("image",image)


import subprocess
result=subprocess.run(['./filter'])  #c++ code 

result=[]
for i in range(10):
    ax=fig.add_subplot(5,4,2*i+2,xticks=[],yticks=[])
    tmp=np.load(f"ansImage{i+1}.npy")
    plt.imshow(tmp)
    result.append(tmp)
    ax.set_title(f"result{i+1}")

plt.show()



########### below is task3 ################




# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

# DataLoader for the Greek data set
training_set_path="greek_train/greek_train"
greek_train = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder( training_set_path,
                                      transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,) ) ] ) ),
    batch_size = 5,
    shuffle = True )

testing_set_path="greek_test/greek_test"
greek_test = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder( testing_set_path,
                                      transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,) ) ] ) ),
    batch_size = 1,
    shuffle=True)


class Model2(nn.Module):
    def __init__(self, model1):
        super(Model2, self).__init__()
        for param in model1.parameters():
            param.requires_grad = False
        self.model_partial1 = nn.Sequential(
            model1.conv1,
            model1.pool1,
        )
        self.model_partial2 = nn.Sequential(
            model1.conv2,
            model1.dropout,
            model1.pool2
        )
        self.Linear1 = model1.fc1
        self.Linear2 = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(self.model_partial1(x))
        x = F.relu(self.model_partial2(x))
        x = x.view(-1, 320)
        x = F.relu(self.Linear1(x))
        x = F.log_softmax(self.Linear2(x))
        return x




model2 = Model2(model)

for name, params in model2.named_parameters():
    print(name, " ", params.size())

criterion = nn.NLLLoss()
optimizer = optim.SGD(model2.parameters(), lr=0.003, momentum=0.9)

epoches = 15
for epoch in range(epoches):
    model2.train()
    acc = 0
    total=0
    correct=0
    for images, labels in greek_train:
        #plt.imshow(images[0].numpy().squeeze())
        # print(images.size())
        optimizer.zero_grad()
        outputs = model2(images)
        _,predict=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predict==labels).sum()
        
        loss = criterion(outputs, labels)   
        loss.backward()
        optimizer.step()
    model2.eval()

    print(f"epoch{epoch} loss: {loss} average_acc: {correct/total}")
    
def imshow(img):
    img=img.numpy().squeeze()
    img=(img*0.1307)+0.3081
    plt.imshow(img)

fig1=plt.figure(figsize=(10,10))

i=0
with torch.no_grad():
    for image,label in greek_test:
        output=model2(image)
        _,predict=torch.max(output.data,1)
        i+=1
        ax=fig1.add_subplot(3,4,i,xticks=[],yticks=[])
        imshow(image)
        ax.set_title(f"predict: {predict.numpy()[0]},actual: {label.numpy()[0]}")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        