"""
Name: [Your Name]
Project 5: Recognition using Deep Networks - Extended Greek Letters
Date: Mar 27, 2025
This script evaluates the performance of the trained Greek letters model.
"""

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from mnist_network import MyNetwork

# Define Greek letters to include
GREEK_LETTERS = ['alpha', 'beta', 'gamma',
                 'delta', 'epsilon', 'zeta', 'eta', 'theta']
NUM_CLASSES = len(GREEK_LETTERS)

# Define paths
MODEL_PATH = "../models/greek_extended_network.pth"
CUSTOM_PATH = "../data/greek_custom_extended"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


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


def load_data_greek_extended(data_path, batch_size=16):
    """
    Loads the extended Greek dataset using a DataLoader with preprocessing transformations.

    Args:
        data_path (str): Path to the dataset directory.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: A DataLoader for the Greek dataset with batching and shuffling.
    """
    # DataLoader for the extended Greek data set
    greek_loader = torch.utils.data.DataLoader(
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
    return greek_loader


def visualize_predictions(network, data_loader, num_examples=16):
    """
    Visualizes model predictions on a batch of examples.

    Args:
        network (torch.nn.Module): The trained neural network model.
        data_loader (DataLoader): DataLoader containing the dataset.
        num_examples (int): Number of examples to visualize.
    """
    network.eval()

    # Get a batch of examples
    examples = iter(data_loader)
    example_data, example_targets = next(examples)

    # Limit to specified number of examples
    example_data = example_data[:num_examples]
    example_targets = example_targets[:num_examples]

    # Get predictions
    with torch.no_grad():
        output = network(example_data)
        predictions = output.argmax(dim=1)

    # Calculate accuracy
    correct = predictions.eq(example_targets).sum().item()
    accuracy = 100. * correct / len(example_data)
    print(f"Batch accuracy: {accuracy:.2f}% ({correct}/{len(example_data)})")

    # Plot examples
    rows = int(np.ceil(num_examples / 4))
    cols = min(4, num_examples)

    fig = plt.figure(figsize=(12, rows * 3))
    for i in range(len(example_data)):
        plt.subplot(rows, cols, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')

        # Mark correct/incorrect predictions
        is_correct = predictions[i] == example_targets[i]
        color = 'green' if is_correct else 'red'

        true_letter = GREEK_LETTERS[example_targets[i]]
        pred_letter = GREEK_LETTERS[predictions[i]]

        plt.title(f"True: {true_letter}\nPred: {pred_letter}",
                  color=color, fontsize=12)
        plt.xticks([])
        plt.yticks([])

    plt.savefig(f'{RESULTS_DIR}/greek_predictions.png')
    plt.show()
    plt.close(fig)


def create_confusion_matrix(network, data_loader):
    """
    Creates and visualizes a confusion matrix for model evaluation.

    Args:
        network (torch.nn.Module): The trained neural network model.
        data_loader (DataLoader): DataLoader containing the dataset.
    """
    # Set network to evaluation mode
    network.eval()

    # Initialize confusion matrix
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    # Collect predictions
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets in data_loader:
            # Forward pass
            outputs = network(data)
            predictions = outputs.argmax(dim=1)

            # Update confusion matrix
            for t, p in zip(targets, predictions):
                confusion_matrix[t.item(), p.item()] += 1

            # Store predictions and targets for later analysis
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Convert to accuracy percentages (row normalization)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_cm = confusion_matrix / row_sums * 100

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(normalized_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    # Add labels
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(GREEK_LETTERS)
    ax.set_yticklabels(GREEK_LETTERS)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = normalized_cm.max() / 2.
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{normalized_cm[i, j]:.1f}%",
                    ha="center", va="center",
                    color="white" if normalized_cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.savefig(f'{RESULTS_DIR}/confusion_matrix.png')
    plt.show()
    plt.close(fig)

    # Calculate overall accuracy
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix) * 100
    print(f"Overall accuracy: {accuracy:.2f}%")

    return confusion_matrix, all_preds, all_targets


def analyze_misclassifications(network, data_loader, max_examples=12):
    """
    Analyzes and visualizes misclassified examples.

    Args:
        network (torch.nn.Module): The trained neural network model.
        data_loader (DataLoader): DataLoader containing the dataset.
        max_examples (int): Maximum number of misclassified examples to show.
    """
    network.eval()

    # Store misclassified examples
    misclassified_data = []
    misclassified_preds = []
    misclassified_targets = []

    with torch.no_grad():
        for data, targets in data_loader:
            # Forward pass
            outputs = network(data)
            predictions = outputs.argmax(dim=1)

            # Find misclassified examples
            incorrect_mask = ~predictions.eq(targets)

            if incorrect_mask.any():
                misclassified_data.append(data[incorrect_mask])
                misclassified_preds.append(predictions[incorrect_mask])
                misclassified_targets.append(targets[incorrect_mask])

            # Break if we have enough examples
            if sum(len(x) for x in misclassified_data) >= max_examples:
                break

    # Prepare data for visualization
    misclassified_data = torch.cat(misclassified_data)[:max_examples]
    misclassified_preds = torch.cat(misclassified_preds)[:max_examples]
    misclassified_targets = torch.cat(misclassified_targets)[:max_examples]

    # Visualize misclassified examples
    num_examples = len(misclassified_data)

    if num_examples == 0:
        print("No misclassified examples found!")
        return

    # Plot examples
    rows = int(np.ceil(num_examples / 4))
    cols = min(4, num_examples)

    fig = plt.figure(figsize=(12, rows * 3))
    for i in range(num_examples):
        plt.subplot(rows, cols, i+1)
        plt.tight_layout()
        plt.imshow(misclassified_data[i][0], cmap='gray', interpolation='none')

        true_letter = GREEK_LETTERS[misclassified_targets[i]]
        pred_letter = GREEK_LETTERS[misclassified_preds[i]]

        plt.title(f"True: {true_letter}\nPred: {pred_letter}",
                  color='red', fontsize=12)
        plt.xticks([])
        plt.yticks([])

    plt.savefig(f'{RESULTS_DIR}/misclassified_examples.png')
    plt.show()
    plt.close(fig)

    # Analyze common misclassifications
    error_pairs = [(GREEK_LETTERS[t.item()], GREEK_LETTERS[p.item()])
                   for t, p in zip(misclassified_targets, misclassified_preds)]

    error_counts = {}
    for true_label, pred_label in error_pairs:
        key = (true_label, pred_label)
        error_counts[key] = error_counts.get(key, 0) + 1

    # Sort by frequency
    sorted_errors = sorted(error_counts.items(),
                           key=lambda x: x[1], reverse=True)

    # Print common misclassifications
    print("\nCommon misclassifications:")
    for (true_label, pred_label), count in sorted_errors:
        print(f"  {true_label} → {pred_label}: {count} instances")


def evaluate_model(model_path, data_path):
    """
    Evaluates the trained model on a dataset.

    Args:
        model_path (str): Path to the trained model file.
        data_path (str): Path to the dataset directory.
    """
    try:
        # Load the model
        network = MyNetwork()
        network.fc2 = nn.Linear(50, NUM_CLASSES)  # Adjust the output layer
        network.load_state_dict(torch.load(
            model_path, map_location=torch.device('cpu')))
        print(f"Loaded model from {model_path}")

        # Load the dataset
        data_loader = load_data_greek_extended(data_path)
        print(
            f"Loaded data from {data_path} with {len(data_loader.dataset)} examples")

        # Visualize predictions
        print("\nVisualizing predictions:")
        visualize_predictions(network, data_loader)

        # Create confusion matrix
        print("\nGenerating confusion matrix:")
        confusion_matrix, all_preds, all_targets = create_confusion_matrix(
            network, data_loader)

        # Analyze misclassifications
        print("\nAnalyzing misclassifications:")
        analyze_misclassifications(network, data_loader)

        # Per-class performance metrics
        print("\nPer-class performance metrics:")
        for i, letter in enumerate(GREEK_LETTERS):
            # Calculate metrics for this class
            true_positives = confusion_matrix[i, i]
            false_positives = np.sum(confusion_matrix[:, i]) - true_positives
            false_negatives = np.sum(confusion_matrix[i, :]) - true_positives

            # Precision, recall, F1 score
            precision = true_positives / \
                (true_positives + false_positives) if (true_positives +
                                                       false_positives) > 0 else 0
            recall = true_positives / \
                (true_positives + false_negatives) if (true_positives +
                                                       false_negatives) > 0 else 0
            f1 = 2 * precision * recall / \
                (precision + recall) if (precision + recall) > 0 else 0

            print(
                f"  {letter}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the model file and dataset directory exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    """
    Main function to evaluate the Greek letters model.
    """
    print("Greek Letters Model Evaluation")
    print("==============================")

    # Check if using custom model path from command line
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = MODEL_PATH

    if len(sys.argv) > 2:
        data_path = sys.argv[2]
    else:
        data_path = CUSTOM_PATH

    # Evaluate the model
    evaluate_model(model_path, data_path)


if __name__ == "__main__":
    main()
