import torch
import torch.nn as nn
import torch.nn.functional as F

# Surce
# https://github.com/python-engineer/pytorchTutorial/blob/master/13_feedforward.py
class Net(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(Net, self).__init__()
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

def train(model, dataloader, epochs, device):
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

def test(model, dataloader, device):
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
        print(f'Accuracy of the network on the 10000 test images: {acc}')
    return loss, acc