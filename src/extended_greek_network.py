"""
Name: [Your Name]
Project 5: Recognition using Deep Networks - Extended Greek Letters
Date: Mar 27, 2025
This file extends the Greek letter classifier to recognize more Greek letters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import os
import numpy as np
from mnist_network import MyNetwork

MODEL_PATH = "../models/mnist_network.pth"
DATA_PATH = "../data/greek_extended"
CUSTOM_PATH = "../data/greek_custom_extended"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define the Greek letters to recognize
GREEK_LETTERS = ['alpha', 'beta', 'gamma',
                 'delta', 'epsilon', 'zeta', 'eta', 'theta']
NUM_CLASSES = len(GREEK_LETTERS)


class GreekTransformExtended(MyNetwork):
    """
    A neural network transformation class that modifies input images before processing.
    This class inherits from MyNetwork and overrides the last fully connected layer 
    to have more output nodes for additional Greek letters.
    """

    def __init__(self):
        super().__init__()

        # Replace the last layer with the appropriate number of nodes
        self.fc2 = nn.Linear(50, NUM_CLASSES)

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


def load_data_greek_extended(data_path, batch_size=5):
    """
    Loads the extended Greek dataset using a DataLoader with preprocessing transformations.

    Args:
        data_path (str): Path to the dataset directory.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: A DataLoader for the Greek dataset with batching and shuffling.
    """
    # DataLoader for the extended Greek data set
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            data_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                GreekTransformExtended(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=batch_size,
        shuffle=True
    )
    return greek_train


def test_custom_data(network, test_loader):
    """
    Tests a neural network on a custom dataset and visualizes predictions.

    Args:
        network (torch.nn.Module): The trained neural network model.
        test_loader (DataLoader): DataLoader for the test dataset.

    Performs inference on a batch, prints the network output, predicted 
    classes, and correct labels, and visualizes the predictions.
    """
    network.eval()

    # Get a batch of test data
    examples = iter(test_loader)
    example_data, example_targets = next(examples)

    # Use all examples in the batch
    batch_size = min(16, len(example_data))
    example_data = example_data[:batch_size]
    example_targets = example_targets[:batch_size]

    with torch.no_grad():
        output = network(example_data)

    # Print the network output values, predicted class, and correct label
    for i in range(batch_size):
        output_values = output[i].numpy()
        predicted_class = output[i].argmax().item()
        correct_label = example_targets[i].item()

        # Print with 2 decimal places
        print(f"Example {i+1}:")
        print(
            f"  Network Output: [{', '.join([f'{val:.2f}' for val in output_values])}]")
        print(
            f"  Predicted Class: {predicted_class} ({GREEK_LETTERS[predicted_class]})")
        print(
            f"  Correct Label: {correct_label} ({GREEK_LETTERS[correct_label]})")

    # Calculate accuracy
    predictions = output.argmax(dim=1)
    correct = predictions.eq(example_targets).sum().item()
    accuracy = 100. * correct / batch_size
    print(f"Batch accuracy: {accuracy:.2f}%")

    # Plot the examples with predictions
    rows = int(np.ceil(batch_size / 4))
    cols = min(4, batch_size)

    fig = plt.figure(figsize=(12, 8))
    for i in range(batch_size):
        plt.subplot(rows, cols, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        correct_str = "✓" if predictions[i] == example_targets[i] else "✗"
        plt.title(f"Pred: {GREEK_LETTERS[predictions[i]]} {correct_str}")
        plt.xticks([])
        plt.yticks([])

    plt.savefig(f'{RESULTS_DIR}/predictions_greek_extended.png')
    plt.show()
    plt.close(fig)


def train_greek_network_extended(network, train_loader, epochs=10):
    """
    Train the neural network on the extended Greek letters dataset.

    Args:
        network: The neural network model
        train_loader: DataLoader for training data
        epochs: Number of training epochs

    Returns:
        Trained network and training accuracies
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
        epoch_losses = []

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)

            loss = criterion(output, target)
            epoch_losses.append(loss.item())

            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)

            # Print progress for larger batches
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # Record average loss for this epoch
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        training_losses.append(avg_epoch_loss)

        # Calculate training accuracy for this epoch
        train_accuracy = 100. * train_correct / train_total
        train_accuracies.append(train_accuracy)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Training Accuracy: {train_accuracy:.2f}%')
        print(f'  Average Loss: {avg_epoch_loss:.6f}')

    # Plot training loss
    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), training_losses,
             'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Likelihood Loss')
    plt.title('Training Loss for Extended Greek Letters')
    plt.legend()
    plt.savefig(f'{RESULTS_DIR}/loss_greek_extended.png')
    plt.show()
    plt.close(fig)

    # Plot training accuracy
    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_accuracies,
             'r-', label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy for Extended Greek Letters')
    plt.legend()
    plt.savefig(f'{RESULTS_DIR}/accuracy_greek_extended.png')
    plt.show()
    plt.close(fig)

    return network, train_accuracies


def visualize_confusion_matrix(network, data_loader):
    """
    Generates and visualizes a confusion matrix for the Greek letters classifier.

    Args:
        network (torch.nn.Module): The trained neural network model.
        data_loader (DataLoader): DataLoader for evaluation.
    """
    # Set the network to evaluation mode
    network.eval()

    # Initialize the confusion matrix
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    # Evaluate on the data
    with torch.no_grad():
        for data, target in data_loader:
            output = network(data)
            pred = output.argmax(dim=1)

            # Update confusion matrix
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.item(), p.item()] += 1

    # Plot the confusion matrix
    fig, ax = plt.figure(figsize=(10, 8)), plt.axes()

    # Create a heatmap
    im = ax.imshow(confusion_matrix, cmap='Blues')

    # Add colorbar
    plt.colorbar(im)

    # Add labels, title and ticks
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(GREEK_LETTERS)
    ax.set_yticklabels(GREEK_LETTERS)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            text = ax.text(j, i, confusion_matrix[i, j],
                           ha="center", va="center", color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black")

    ax.set_title("Confusion Matrix for Greek Letters")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()

    plt.savefig(f'{RESULTS_DIR}/confusion_matrix_greek.png')
    plt.show()
    plt.close(fig)


def main():
    """
    Loads a pretrained neural network, modifies its output layer, and trains it 
    on the extended Greek dataset before testing it on a custom dataset.
    """
    # Load the network
    network = MyNetwork()
    network.load_state_dict(torch.load(
        MODEL_PATH, map_location=torch.device('cpu')))

    # Freeze the parameters
    for param in network.parameters():
        param.requires_grad = False

    # Replace with appropriate number of outputs
    network.fc2 = nn.Linear(50, NUM_CLASSES)
    print(network)

    # Load the training data
    train_loader = load_data_greek_extended(DATA_PATH)
    print(f"Loaded training data with {len(train_loader.dataset)} images")

    # Train the network
    network, train_accuracies = train_greek_network_extended(
        network, train_loader, epochs=15)

    # Save the trained model
    os.makedirs('models', exist_ok=True)
    torch.save(network.state_dict(), '../models/greek_extended_network.pth')
    print("Network saved to ../models/greek_extended_network.pth")

    # Classify custom dataset if available
    if os.path.exists(CUSTOM_PATH):
        test_loader = load_data_greek_extended(CUSTOM_PATH)
        print(
            f"Loaded custom test data with {len(test_loader.dataset)} images")
        test_custom_data(network, test_loader)

        # Generate confusion matrix
        visualize_confusion_matrix(network, test_loader)
    else:
        print(
            f"Custom dataset path {CUSTOM_PATH} not found. Skipping evaluation on custom data.")


if __name__ == "__main__":
    main()
