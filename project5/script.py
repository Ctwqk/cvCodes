import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.onnx

batchSize=64
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True)

# Download and load the test data
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(1,10,kernel_size=5)
        self.pool1=nn.MaxPool2d(2,2)
        
        self.conv2=nn.Conv2d(10,20,kernel_size=5)
        self.dropout=nn.Dropout(p=0.5)
        self.pool2=nn.MaxPool2d(2,2)
        
        self.fc1=nn.Linear(320,50)
        self.fc2=nn.Linear(50,10)

    def forward(self, x):
        x=F.relu(self.pool1(self.conv1(x)))
        x=F.relu(self.pool2(self.dropout(self.conv2(x))))
        x=x.view(-1,320)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return F.log_softmax(x, dim=1)

def train():
    net = Net()

    import torch.optim as optim

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

    evaluate_every=1000
    epoches=5
    seen_examples=0

    examples=[]
    losses=[]
    losses_test=[]

    for epoch in range(epoches):  # loop over the dataset multiple times
        running_loss = 0.0
        
        for images, labels in trainloader:
            # zero the parameter gradients
            optimizer.zero_grad()
            seen_examples+=batchSize
            
            #seen_examples+=batchSize;
            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            
            if seen_examples%evaluate_every==0 or seen_examples==epoch*len(trainloader):
                net.eval()
                total_loss=0.0
                total=0
                
                with torch.no_grad():
                    for images, labels in trainloader:
                        outputs=net(images)
                        loss=criterion(outputs,labels)
                        total_loss+= loss.item() * images.size(0)
                        total +=images.size(0)
                    
                average_loss = total_loss / total
                examples.append(seen_examples)
                losses.append(average_loss)
                
                total_loss=0.0
                total=0
                with torch.no_grad():
                    for images, labels in testloader:
                        outputs=net(images)
                        loss=criterion(outputs,labels)
                        total_loss+= loss.item() * images.size(0)
                        total +=images.size(0)
                    
                average_loss = total_loss / total
                losses_test.append(average_loss)
                net.train()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')


    plt.plot(examples,losses,color='blue')
    plt.plot(examples,losses_test,color='red')
    plt.show()

    
    torch.save(net.state_dict(),"mnist_model_wieghts.pth")
    for inputs,_ in trainloader:
        dummy_input=inputs
        break
    net.eval()
    torch.onnx.export(net,dummy_input,"mnist_model.onnx",export_params=True, \
                      opset_version=10, \
                      do_constant_folding=True, \
                      input_names=['input'], \
                      output_names=['output'], \
                      dynamic_axes={'input': {0: 'batch_size'}, \
                                    'output': {0: 'batch_size'}})

if __name__=="__main__":
    train()
    





