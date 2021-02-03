import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np

transform=transforms.ToTensor()


Train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
Test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

batch_size = 20
num_workers = 0

train_loader = torch.utils.data.DataLoader(Train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(Test_data, batch_size=batch_size, num_workers=num_workers)

import torch.nn as nn
import torch.nn.functional as F
p=0.3
class Net(nn.Module):
    def __init__(self, p):
        super(Net,self).__init__()
        self.bn2 = nn.BatchNorm2d(1)
        self.conv1= nn.Conv2d(1,6,3)
        self.conv2= nn.Conv2d(6,16,3)
        self.conv3= nn.Conv2d(16,26,3)
        self.fc0= nn.Linear(26*9*9,16*5*5)
        self.fc1= nn.Linear(16*5*5,120)
        self.fc2= nn.Linear(120,84)
        self.fc3= nn.Linear(84,10)
        self.dropout1 = nn.Dropout(p)




    def forward(self,x):
        x=self.bn2(x)
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=F.relu(self.fc0(x.view(-1,self.num_flat_features(x))))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.dropout1(self.fc3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



net=Net(p)
print(net)

n_epochs=30
net.train()
criterion= nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.05)
def to_one_hot_vector(num_class, label):
    b = torch.zeros((label.shape[0], num_class))
    b[torch.arange(label.shape[0]), label] = 1

    return b

for n in range(n_epochs):

    train_loss=0.0
    for x,y in train_loader:
        optimizer.zero_grad()
        output=net(x)
        labels_one_hot = to_one_hot_vector(10, y)
        labels_one_hot = torch.Tensor(labels_one_hot)
        labels_one_hot = labels_one_hot.type(torch.FloatTensor)

        loss = criterion(output, labels_one_hot)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        n + 1,
        train_loss
    ))

test_loss = 0.0
class_correct = list(0 for i in range(10))
class_total = list(0 for i in range(10))

net.eval().cpu()

for data, target in test_loader:
    output = net(data)

    labels_one_hot = to_one_hot_vector(10, target)
    labels_one_hot = torch.Tensor(labels_one_hot)
    labels_one_hot = labels_one_hot.type(torch.FloatTensor)

    loss = criterion(output, labels_one_hot)

    test_loss += loss.item()*data.size(0)

    _, pred = torch.max(output, 1)

    correct = (pred == target.view_as(pred)).squeeze()

    for i in range(batch_size):
        label = target[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1


test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)')

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))






