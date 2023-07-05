from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.nn import functional
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

# print(torch.__version__)

""" Use cuda if available """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" Network classes """
class NN(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, n_classes)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(in_features=16*7*7, out_features=n_classes)

    def forward(self, x):
        x = functional.relu(self.conv1(x))
        x = self.pool(x)
        x = functional.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


""" Calculate accuracy """
def evaluate(loader, model):
    n_accurate = 0
    n_samples = 0
    model.eval()  # set eval mode

    with torch.no_grad():  # no need for gradient calculation during evaluation
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            n_accurate += (predictions==y).sum()
            n_samples += predictions.shape[0]

        acc = float(n_accurate) / float(n_samples)
        print(f'{n_accurate} / {n_samples} with acc: {acc:.3f}')
    model.train()  # set train mode


def visualize_samples():
    train_dataset = datasets.MNIST(root='./datasets/', train=True, transform=transforms.ToTensor(), download=True)
    figure = plt.figure(figsize=(6, 6))
    cols, rows = 3, 3
    for i in range(1, rows*cols+1):
        rand_idx = torch.randint(len(train_dataset), size=(1,)).item()
        image, label = train_dataset[rand_idx]
        image = image.reshape(28, 28, 1)
        figure.add_subplot(rows, cols, i)
        plt.title(f'label: {label}')
        plt.imshow(image, cmap='gray')
        plt.axis("off")
    plt.show()


def main_fc():
    """ Hyperparameters """
    input_size = 784
    n_classes = 10
    learning_rate = 0.001
    batch_size = 64
    n_epochs = 1

    """ Data loading """
    train_dataset = datasets.MNIST(root='./datasets/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./datasets/', train=False, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    """ Network Initialization """
    model = NN(input_size=input_size, n_classes=n_classes)

    """ Loss and Optimizer """
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('\n\n TRAINING STARTS\n')
    """ Training loop """
    for epoch in tqdm(range(n_epochs)):
        for batch_id, (data, target) in enumerate(train_loader):
            # Use cuda if available
            data = data.to(device=device)
            target = target.to(device=device)

            # Reshape [M x 1 x 28 x 28] to [M x 784]
            data = data.reshape(data.shape[0], -1)

            # Forward
            scores = model(data)
            loss = loss_function(scores, target)

            # Backward
            optimizer.zero_grad()  # to remove the gradients from previous batch, if any
            loss.backward()

            # SGD or Adam (update weights)
            optimizer.step()
    print('\n\n TRAINING ENDS\n')

    evaluate(train_loader, model)
    evaluate(test_loader, model)


def main_cnn():
    """ Hyperparameters """
    in_channels = 1
    n_classes = 10
    learning_rate = 0.001
    batch_size = 64
    n_epochs = 1

    """ Data loading """
    train_dataset = datasets.MNIST(root='./datasets/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./datasets/', train=False, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    """ Network Initialization """
    model = CNN(in_channels=in_channels, n_classes=n_classes)

    """ Loss and Optimizer """
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('\n\n TRAINING STARTS\n')
    """ Training loop """
    for epoch in tqdm(range(n_epochs)):
        for batch_id, (data, target) in enumerate(train_loader):
            # Use cuda if available
            data = data.to(device=device)
            target = target.to(device=device)

            # Forward
            scores = model(data)
            loss = loss_function(scores, target)

            # Backward
            optimizer.zero_grad()  # to remove the gradients from previous batch, if any
            loss.backward()

            # SGD or Adam (update weights)
            optimizer.step()
    print('\n\n TRAINING ENDS\n')

    evaluate(train_loader, model)
    evaluate(test_loader, model)

if __name__ == '__main__':
    # visualize_samples()
    main_cnn()