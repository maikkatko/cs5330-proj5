"""
Name: Seunghan, Lee / Maik, Katko
Project 5: Recognition using Deep Networks - Experiment
Date: Mar 23rd, 2025
This file includes MNIST network architecture experiment for fashion data.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import pandas as pd
from itertools import product
from mnist_network import load_data

# Check if CUDA is available and set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

DATA_PATH = "data/MNIST"
FASHION_DATA_PATH = "data/FashionMNIST"
RESULTS_DIR = "./experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class ExperimentNetwork(nn.Module):
    """
    A flexible neural network that can be configured with different architectures
    for experimentation.
    """

    def __init__(self,
                 # Number of filters in each conv layer
                 conv_filters=[10, 20],
                 # Size of filters in each conv layer
                 filter_sizes=[5, 5],
                 fc_nodes=50,                  # Number of nodes in fully connected layer
                 dropout_rate=0.25,            # Dropout rate
                 pool_size=2,                  # Pooling size
                 activation_fn=nn.ReLU(),      # Activation function
                 fc_dropout=False):            # Whether to use dropout after fully connected
        super(ExperimentNetwork, self).__init__()

        self.layers = nn.ModuleList()
        self.activation_fn = activation_fn

        # First conv layer
        self.conv1 = nn.Conv2d(1, conv_filters[0], kernel_size=filter_sizes[0])
        self.layers.append(self.conv1)

        # First pooling layer
        self.pool1 = nn.MaxPool2d(pool_size)
        self.layers.append(self.pool1)

        # Second conv layer with dropout
        self.conv2 = nn.Conv2d(
            conv_filters[0], conv_filters[1], kernel_size=filter_sizes[1])
        self.conv_dropout = nn.Dropout2d(dropout_rate)
        self.layers.append(self.conv2)
        self.layers.append(self.conv_dropout)

        # Second pooling layer
        self.pool2 = nn.MaxPool2d(pool_size)
        self.layers.append(self.pool2)

        # Calculate the flattened size
        # Start with 28x28 input
        first_conv_output = 28 - filter_sizes[0] + 1  # Output after first conv
        first_pool_output = first_conv_output // pool_size  # Output after first pooling
        second_conv_output = first_pool_output - \
            filter_sizes[1] + 1  # Output after second conv
        second_pool_output = second_conv_output // pool_size  # Output after second pooling

        # Flattened size
        self.flattened_size = conv_filters[1] * \
            second_pool_output * second_pool_output

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, fc_nodes)
        self.use_fc_dropout = fc_dropout
        self.fc_dropout_layer = nn.Dropout(
            dropout_rate)  # Renamed to be more explicit
        self.fc2 = nn.Linear(fc_nodes, 10)

    # computes a forward pass for the network
    def forward(self, x):
        # Apply convolutional and pooling layers
        for layer in self.layers:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                x = self.activation_fn(layer(x))
            else:
                x = layer(x)

        # Flatten the tensor
        x = x.view(-1, self.flattened_size)

        # Apply first fully connected layer with activation
        x = self.activation_fn(self.fc1(x))

        # Apply dropout after fully connected if specified
        if self.use_fc_dropout:
            x = self.fc_dropout_layer(x)  # Use the renamed layer

        # Apply final fully connected layer
        x = self.fc2(x)

        # Apply log_softmax function
        return torch.nn.functional.log_softmax(x, dim=1)


def load_fashion_data(batch_size=64):
    """
    Load the Fashion MNIST dataset.
    Returns the training and test dataloaders.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Fashion MNIST mean and std
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    # Load training data
    train_dataset = torchvision.datasets.FashionMNIST(
        root=FASHION_DATA_PATH,
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
    test_dataset = torchvision.datasets.FashionMNIST(
        root=FASHION_DATA_PATH,
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


def train_and_evaluate(network, train_loader, test_loader, epochs=5, batch_size=64, learning_rate=0.01):
    """
    Train and evaluate the neural network.

    Args:
        network: The neural network model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer

    Returns:
        Dictionary with training metrics
    """
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.5)
    criterion = nn.NLLLoss()

    # For recording metrics
    train_accuracies = []
    test_accuracies = []
    training_times = []

    start_time_total = time.time()

    for epoch in range(epochs):
        # Training
        network.train()
        train_correct = 0
        train_total = 0

        epoch_start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        training_times.append(epoch_time)

        train_accuracy = 100. * train_correct / train_total
        train_accuracies.append(train_accuracy)

        # Testing
        network.eval()
        test_correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()

        test_accuracy = 100. * test_correct / len(test_loader.dataset)
        test_accuracies.append(test_accuracy)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Training Accuracy: {train_accuracy:.2f}%')
        print(f'  Testing Accuracy: {test_accuracy:.2f}%')
        print(f'  Epoch Time: {epoch_time:.2f}s')

    total_time = time.time() - start_time_total

    # Return metrics
    return {
        'final_train_accuracy': train_accuracies[-1],
        'final_test_accuracy': test_accuracies[-1],
        'max_test_accuracy': max(test_accuracies),
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'avg_epoch_time': sum(training_times) / len(training_times),
        'total_time': total_time
    }


def run_experiment(experiment_config, dataset='mnist'):
    """
    Run a single experiment with the given configuration.

    Args:
        experiment_config: Dictionary with experiment parameters
        dataset: 'mnist' or 'fashion' dataset to use

    Returns:
        Dictionary with experiment results
    """
    # Load the appropriate dataset
    if dataset == 'mnist':
        train_loader, test_loader = load_data(
            batch_size=experiment_config.get('batch_size', 64))
    else:
        train_loader, test_loader = load_fashion_data(
            batch_size=experiment_config.get('batch_size', 64))

    # Create the network with the specified configuration
    network = ExperimentNetwork(
        conv_filters=experiment_config.get('conv_filters', [10, 20]),
        filter_sizes=experiment_config.get('filter_sizes', [5, 5]),
        fc_nodes=experiment_config.get('fc_nodes', 50),
        dropout_rate=experiment_config.get('dropout_rate', 0.25),
        pool_size=experiment_config.get('pool_size', 2),
        activation_fn=experiment_config.get('activation_fn', nn.ReLU()),
        fc_dropout=experiment_config.get('fc_dropout', False)
    )

    # Train and evaluate
    results = train_and_evaluate(
        network,
        train_loader,
        test_loader,
        epochs=experiment_config.get('epochs', 5),
        batch_size=experiment_config.get('batch_size', 64),
        learning_rate=experiment_config.get('learning_rate', 0.01)
    )

    # Add the configuration to the results
    results.update(experiment_config)

    return results


def run_experiments(dimensions, dataset='mnist'):
    """
    Run a series of experiments by varying different dimensions.

    Args:
        dimensions: Dictionary where keys are dimension names and values are lists of values to test
        dataset: 'mnist' or 'fashion' dataset to use

    Returns:
        List of dictionaries with experiment results
    """
    results = []

    # Create baseline configuration
    baseline_config = {
        'conv_filters': [10, 20],
        'filter_sizes': [5, 5],
        'fc_nodes': 50,
        'dropout_rate': 0.25,
        'pool_size': 2,
        'activation_fn': nn.ReLU(),
        'fc_dropout': False,
        'epochs': 5,
        'batch_size': 64,
        'learning_rate': 0.01
    }

    # Run baseline experiment
    print("Running baseline experiment...")
    baseline_result = run_experiment(baseline_config, dataset)
    baseline_result['experiment_type'] = 'baseline'
    results.append(baseline_result)

    # For each dimension, run experiments
    for dim_name, dim_values in dimensions.items():
        print(f"Exploring dimension: {dim_name}")
        for value in dim_values:
            # Create a copy of the baseline configuration
            config = baseline_config.copy()

            # Update the dimension being tested
            if dim_name == 'conv_filters':
                # Maintain the ratio between layers
                config[dim_name] = [value, value * 2]
            elif dim_name == 'filter_sizes':
                config[dim_name] = [value, value]  # Same size for both layers
            elif dim_name == 'activation_fn':
                if value == 'relu':
                    config[dim_name] = nn.ReLU()
                elif value == 'tanh':
                    config[dim_name] = nn.Tanh()
                elif value == 'sigmoid':
                    config[dim_name] = nn.Sigmoid()
            else:
                config[dim_name] = value

            # Run the experiment
            print(f"  Testing {dim_name} = {value}")
            result = run_experiment(config, dataset)
            result['experiment_type'] = dim_name
            result['dimension_value'] = value
            results.append(result)

    return results


def analyze_results(results):
    """
    Analyze experiment results and create visualizations.

    Args:
        results: List of dictionaries with experiment results
    """
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)

    # Save raw results to CSV
    df.to_csv(os.path.join(RESULTS_DIR, 'experiment_results.csv'), index=False)

    # Group results by experiment type
    grouped = df.groupby('experiment_type')

    # Plot the results for each dimension
    for exp_type, group in grouped:
        if exp_type == 'baseline':
            continue

        # Sort by dimension value
        if exp_type in ['conv_filters', 'filter_sizes', 'fc_nodes', 'pool_size', 'epochs', 'batch_size']:
            group = group.sort_values('dimension_value')

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot test accuracy
        plt.subplot(1, 2, 1)
        plt.plot(group['dimension_value'], group['final_test_accuracy'], 'b-o')
        plt.axhline(y=df[df['experiment_type'] == 'baseline']['final_test_accuracy'].values[0],
                    color='r', linestyle='--', label='Baseline')
        plt.xlabel(exp_type)
        plt.ylabel('Test Accuracy (%)')
        plt.title(f'Effect of {exp_type} on Test Accuracy')
        plt.grid(True)
        plt.legend()

        # Plot training time
        plt.subplot(1, 2, 2)
        plt.plot(group['dimension_value'], group['avg_epoch_time'], 'g-o')
        plt.axhline(y=df[df['experiment_type'] == 'baseline']['avg_epoch_time'].values[0],
                    color='r', linestyle='--', label='Baseline')
        plt.xlabel(exp_type)
        plt.ylabel('Avg Epoch Time (s)')
        plt.title(f'Effect of {exp_type} on Training Time')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'{exp_type}_results.png'))
        plt.close()

    # Create summary table
    summary = df.groupby('experiment_type').agg({
        'final_test_accuracy': ['mean', 'max'],
        'avg_epoch_time': 'mean',
        'total_time': 'mean'
    }).reset_index()

    summary.columns = ['Dimension', 'Mean Test Accuracy',
                       'Max Test Accuracy', 'Mean Epoch Time', 'Mean Total Time']
    summary.to_csv(os.path.join(
        RESULTS_DIR, 'experiment_summary.csv'), index=False)

    return summary


def main(argv):
    """
    Main function for the MNIST network experiment.

    Args:
        argv: Command line arguments
    """
    # Parse command-line arguments
    dataset = 'mnist'
    if len(argv) > 1 and argv[1] == 'fashion':
        dataset = 'fashion'
        print("Using Fashion MNIST dataset")
    else:
        print("Using MNIST digits dataset")

    # Define experiment dimensions
    dimensions = {
        # Number of filters in first conv layer (second layer is 2x)
        'conv_filters': [5, 10, 15, 20],
        'filter_sizes': [3, 5, 7],        # Size of conv filters
        # Number of nodes in fully connected layer
        'fc_nodes': [20, 50, 100, 200],
        'dropout_rate': [0.1, 0.25, 0.5],  # Dropout rate
        'pool_size': [2, 3, 4],           # Size of pooling window
        'activation_fn': ['relu', 'tanh', 'sigmoid'],  # Activation function
        # Whether to use dropout after fully connected layer
        'fc_dropout': [True, False],
        'epochs': [3, 5, 7, 10],           # Number of training epochs
        'batch_size': [32, 64, 128, 256],  # Batch size
    }

    # Choose three dimensions to explore
    selected_dimensions = {
        'conv_filters': dimensions['conv_filters'],      # Number of filters
        'dropout_rate': dimensions['dropout_rate'],      # Dropout rate
        'batch_size': dimensions['batch_size']           # Batch size
    }

    # Run experiments
    results = run_experiments(selected_dimensions, dataset)

    # Analyze results
    summary = analyze_results(results)
    print("\nExperiment Summary:")
    print(summary)

    print(f"\nDetailed results saved to {RESULTS_DIR}/experiment_results.csv")
    print(f"Plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main(sys.argv)
