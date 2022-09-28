import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Source: https://github.com/python-engineer/pytorchTutorial/blob/master/13_feedforward.py
class MnistNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(MnistNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

def train_mnist(model, dataloader, epochs, device):
    n_total_steps = len(dataloader)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(dataloader):  
            # origin shape: [100, 1, 28, 28]
            # resized: [100, 784]
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

def test_mnist(model, dataloader, device):
    loss=0
    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in dataloader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = n_correct / n_samples
    return loss, acc


# Source: https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch
class ToyNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=1):
        super(ToyNN, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.sigmoid(self.layer_2(x))
        return x

def train_toy(model, dataloader, epochs, device):
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_values = []
    for epoch in range(epochs):
        for X, y in dataloader:
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward + backward + optimize
            pred = model(X)
            loss = loss_fn(pred, y.unsqueeze(-1))
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()

def test_toy(model, dataloader, device):
    y_pred = []
    y_test = []
    total = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            outputs = model(X)
            predicted = np.where(outputs < 0.5, 0, 1)
            predicted = list(itertools.chain(*predicted))
            y_pred.append(predicted)
            y_test.append(y)
            total += y.size(0)
            correct += (predicted == y.numpy()).sum().item()
    # loss, accuracy
    return 0, correct / total