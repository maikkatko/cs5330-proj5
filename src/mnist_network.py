# Name: Seunghan, Lee / Maik, Katko
# Project 5: Recognition using Deep Networks
# Date: Mar 21st, 2025
# MNIST digit recognition using PyTorch

# import statements
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_PATH = "../data/MNIST"

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # A convolution layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # A max pooling layer with a 2x2 window and a ReLU function applied
        self.pool = nn.MaxPool2d(2)
        # A convolution layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # A dropout layer with a dropout rate of your choice
        self.dropout = nn.Dropout2d(0.25)  # Can be adjusted (5-50%)
        # Flattening happens in forward method
        # A fully connected Linear layer with 50 nodes and a ReLU function
        # 320 = 20 channels * 4 * 4 (size after conv and pooling)
        self.fc1 = nn.Linear(320, 50)
        # A final fully connected Linear layer with 10 nodes
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    def forward(self, x):
        # Apply first convolution followed by pooling and ReLU
        x = self.pool(torch.relu(self.conv1(x)))
        # Apply second convolution with dropout, followed by pooling and ReLU
        x = self.pool(torch.relu(self.dropout(self.conv2(x))))
        # Flatten the tensor
        x = x.view(-1, 320)
        # Apply first fully connected layer with ReLU
        x = torch.relu(self.fc1(x))
        # Apply final fully connected layer
        x = self.fc2(x)
        # Apply log_softmax function
        return torch.nn.functional.log_softmax(x, dim=1)

def load_data(batch_size=64):
    """
    Load the MNIST digit dataset.
    Returns the training and test dataloaders.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load training data
    train_dataset = torchvision.datasets.MNIST(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Load test data
    test_dataset = torchvision.datasets.MNIST(
        root=DATA_PATH,
        train=False,
        download=True,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False
    )

    return train_loader, test_loader


def visualize_examples(data_loader, num_examples=6):
    """
    Visualize the first six examples from the dataset.

    Args:
        data_loader: The data loader containing the dataset
        num_examples: Number of examples to visualize
    """
    # Get the first batch
    examples = iter(data_loader)
    example_data, example_targets = next(examples)

    # Plot the examples
    fig = plt.figure(figsize=(10, 6))
    for i in range(num_examples):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title(f"Ground Truth: {example_targets[i]}")
        plt.xticks([])
        plt.yticks([])

    # Save the figure
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/examples.png')
    plt.show()
    plt.close(fig)


def train_network(network, train_loader, test_loader, epochs=5):
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
    test_accuracies = []
    training_losses = []
    test_losses = []

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

        # Testing
        network.eval()
        test_loss = 0
        test_correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        test_accuracy = 100. * test_correct / len(test_loader.dataset)
        test_accuracies.append(test_accuracy)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Test Loss: {test_loss:.4f}')
        print(f'  Training Accuracy: {train_accuracy:.2f}%')
        print(f'  Testing Accuracy: {test_accuracy:.2f}%')

    # Plot training and test loss
    fig = plt.figure(figsize=(10, 6))
    print(f"len_test: {len(test_losses)}")
    for i in range(len(test_losses)):
        print(f"This is the test losses: {test_losses[i]}")
    plt.plot(range(1, len(training_losses) + 1), training_losses,
             'b-', label='Training Losses')
    plt.scatter(range(1, len(training_losses) + 1, int(len(training_losses) / epochs)), test_losses, color='r', s=50) 
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/loss.png')
    plt.show()
    plt.close(fig)

    # Plot training and test accuracy
    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), test_accuracies,
             'b-', label='Testing Accuracy')
    plt.plot(range(1, epochs+1), train_accuracies,
             'r-', label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.savefig('./results/accuracy.png')
    plt.show()
    plt.close(fig)

    return network, train_accuracies, test_accuracies

def save_network(network, filename='../models/mnist_network.pth'):
    """
    Save the trained network to a file.

    Args:
        network: Trained neural network
        filename: Path to save the network
    """
    os.makedirs('../models', exist_ok=True)
    torch.save(network.state_dict(), filename)
    print(f"Network saved to {filename}")

def test_network(network, test_loader, visualize=True):
    """
    Test the network on the first 10 examples of the test set.

    Args:
        network: Trained neural network
        test_loader: DataLoader for test data
    """
    network.eval()

    # Get the first batch
    examples = iter(test_loader)
    example_data, example_targets = next(examples)
    example_data = example_data[:10]
    example_targets = example_targets[:10]

    with torch.no_grad():
        output = network(example_data)

    # Print the network output values, predicted class, and correct label
    for i in range(10):
        output_values = output[i].numpy()
        predicted_class = output[i].argmax().item()
        correct_label = example_targets[i].item()

        # Print with 2 decimal places
        print(f"Example {i+1}:")
        print(
            f"  Network Output: [{', '.join([f'{val:.2f}' for val in output_values])}]")
        print(f"  Predicted Class: {predicted_class}")
        print(f"  Correct Label: {correct_label}")

    # Plot the first 9 examples with predictions
    if visualize:
        fig = plt.figure(figsize=(12, 8))
        for i in range(9):
            plt.subplot(3, 3, i+1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
            plt.title(f"Prediction: {output[i].argmax().item()}")
            plt.xticks([])
            plt.yticks([])
        os.makedirs('./results', exist_ok=True)
        plt.savefig('./results/predictions.png')
        plt.show()
        plt.close(fig)

def main(argv):
    """
    Main function for the MNIST digit recognition task.

    Args:
        argv: Command line arguments
    """
    # Load data
    train_loader, test_loader = load_data()

    # Visualize examples
    # visualize_examples(test_loader)

    # Create network
    network = MyNetwork()
    # print(network)

    # Train network
    network, _, _ = train_network(
        network, train_loader, test_loader)

    # Save network
    # save_network(network)

    # Test network
    # test_network(network, test_loader)

if __name__ == "__main__":
    main(sys.argv)