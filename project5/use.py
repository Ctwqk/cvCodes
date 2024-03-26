import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from script import Net 


model=Net()
model.load_state_dict(torch.load("mnist_model_wieghts.pth"))


model.eval()


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)


def imshow(img):
    img=img.numpy().squeeze()
    img=(img*0.5)+0.5
    plt.imshow(img,cmap='gray')

fig=plt.figure(figsize=(10,10))

imageSet=[]
label=[]

with torch.no_grad():
    for images,labels in testloader:
        outputs=model(images)
        _,preds=torch.max(outputs,1)
        label=list(zip(preds,labels))
        imageSet=images
        break
for i in range(10):
    print(f"{i+1}: predict: {label[i][0]}, actual: {label[i][1]}")        

for i in range(9):
    ax=fig.add_subplot(3,3,i+1,xticks=[],yticks=[])
    imshow(imageSet[i])
    ax.set_title(f"predict: {label[i][0]}, actual: {label[i][1]}")
        
        