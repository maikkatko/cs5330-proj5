import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from mnist_network import MyNetwork, test_network

MODEL_PATH = "../models/mnist_network.pth"
DATA_PATH = "../data/greek_train"

class GreekTransform(MyNetwork):
    def __init__(self):
        super().__init__()
        
        # replace the last layer with the 3 nodes
        self.fc2 = nn.Linear(50, 3)

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

def load_data_greek():
    # DataLoader for the Greek data set
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder( DATA_PATH,
                                          transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                       GreekTransform(),
                                                                                       torchvision.transforms.Normalize(
                                                                                           (0.1307,), (0.3081,) ) ] ) ),
        batch_size = 5,
        shuffle = True )
    return greek_train

def train_greek_network(network, train_loader, epochs=5):
    """
    Train the neural network.

    Args:
        network: The neural network model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        epochs: Number of training epochs

    Returns:
        Trained network and training/testing accuracies
    """
    optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.NLLLoss()

    train_accuracies = []
    training_losses = []

    for epoch in range(epochs):
        # Training
        network.train()
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            
            loss = criterion(output, target)
            training_losses.append(loss.item())
            
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)

            # Print progress
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        train_accuracy = 100. * train_correct / train_total
        train_accuracies.append(train_accuracy)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Training Accuracy: {train_accuracy:.2f}%')

    # Plot training loss
    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_losses) + 1), training_losses,
             'b-', label='Training Losses')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/loss_greek.png')
    plt.show()
    plt.close(fig)

    # Plot training accuracy
    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_accuracies,
             'r-', label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.savefig('./results/accuracy_greek.png')
    plt.show()
    plt.close(fig)

    return network, train_accuracies

def main():

    # Load the network
    network = MyNetwork()
    network.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

    # Freeze the params
    for param in network.parameters():
        param.requires_grad = False

    # Replace with 3 output
    network.fc2 = nn.Linear(50, 3)
    print(network)

    # Load the datat
    # train_loader = load_data_greek()
    # print(train_loader)

    # train_greek_network(network, train_loader, 5)

if __name__ == "__main__":
    main()