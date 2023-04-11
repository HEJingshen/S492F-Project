import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
import torchvision
from torchvision import transforms
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import os

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Obtain dataset info from csv file
path = os.getcwd()
filename = path + '/DATA/number_of_samples.csv'
df = pd.read_csv(filename)
className = np.array(df['Name of class'])
filesNumber = np.array(df['Number of files'])
classNumber = len(className)

# Set paths
data_path = path + '/DATA'
train_path = data_path + '/train'
val_path = data_path + '/val'

# Get data from path and resize to 224x224
def get_data(_path):
    data = []
    labels = []
    for i in range(classNumber):
        path = _path + '/' + className[i] + '/'
        images = os.listdir(path)
        for a in images:
            img = Image.open(os.path.join(path, a))
            img = img.resize((224, 224), Image.ANTIALIAS)
            data.append(img)
            labels.append(i)
    return data, labels

data, labels = get_data(train_path)
val_data, val_labels = get_data(val_path)

# Convert data to tensor and normalize
transforms = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, trans):
        self.data = data
        self.labels = labels
        self.trans = trans

    def __getitem__(self, index):
        return self.trans(self.data[index]), self.labels[index]
    
    def __len__(self):
        return len(self.data)
    
train_dataset = MyDataset(data, labels, transforms)
val_dataset = MyDataset(val_data, val_labels, transforms)

# Create model (LeNet-5)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train model
learning_rate = 0.0009575
batch_size = 32
num_epochs = 50

train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

model = Net().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# Learning rate scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.35, patience=125, verbose=1, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

def train(model, train_loader, device):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss_ = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss_.backward()
        total_loss += loss_.item()
        optimizer.step()
        #scheduler.step(total_loss)
        
        correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
        total_samples += labels.size(0)
    print('Train Accuracy of the model on the {} train images: {} %'.format(total_samples, correct / total_samples))
    return total_loss / len(train_loader), correct / total_samples

def val(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        correct = 0
        total_samples = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
            loss_ = criterion(outputs, labels)
            total_loss += loss_.item()
            total_samples += labels.size(0)
            
        print('Val Accuracy of the model on the {} val images: {} %'.format(total_samples, correct / total_samples))
        return total_loss / len(val_loader), correct / total_samples

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, device=device)
    val_loss, val_acc = val(model, val_loader, device=device)

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    torch.save(model.state_dict(), path + '/model/model.pth')

# Draw graph
def draw_graph(x, train_Y, val_Y, ylabel, path):
    plt.plot(x, train_Y, label='Train_' + ylabel, linewidth=1.5)
    plt.plot(x, val_Y, label='Val_' + ylabel, linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    plt.savefig(path + '/model/' + ylabel + '.png')

x = np.linspace(0, len(train_loss_list), len(train_loss_list))
draw_graph(x, train_loss_list, val_loss_list, 'Loss', path)
draw_graph(x, train_acc_list, val_acc_list, 'Accuracy', path)