import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from mnist_network import MyNetwork, load_data, test_network

MODEL_PATH = "../models/mnist_network.pth"
CUSTOM_PATH = "../data/MNIST_custom"

def load_custom_data(custom_data_path):
    # DataLoader for the Greek data set
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder( custom_data_path,
                                          transform = torchvision.transforms.Compose( [transforms.Resize((28, 28)),
                                                                                       transforms.Grayscale(num_output_channels=1),
                                                                                       torchvision.transforms.ToTensor(),
                                                                                       torchvision.transforms.Normalize(
                                                                                           (0.1307,), (0.3081,) ) ] ) ),
        batch_size = 10,
        shuffle = True )
    return greek_train

def test_custom_data(network, test_loader):

    network.eval()

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
    fig = plt.figure(figsize=(12, 8))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title(f"Prediction: {output[i].argmax().item()}")
        plt.xticks([])
        plt.yticks([])
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/predictions_custom.png')
    plt.show()
    plt.close(fig)

def main():
    _, test_loader = load_data()
    network = MyNetwork()
    network.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    # test_network(network, test_loader)

    # test on the custom data
    test_loader = load_custom_data(CUSTOM_PATH)
    test_custom_data(network, test_loader)

if __name__ == "__main__":
    main()