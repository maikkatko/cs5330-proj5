"""
Name: Seunghan, Lee / Maik, Katko
Project 5: Recognition using Deep Networks
Date: Mar 21st, 2025
This file contains the extension for visualizing weights for the resnet 50 network.
"""

import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
from mnist_network import load_data

MODEL_PATH = "../models/mnist_network.pth"

def load_data():
    """
    Loads the CIFAR-10 test dataset and prepares it for loading in batches.
    
    The function applies a transformation pipeline that converts the images to 
    tensors and normalizes them using a mean and standard deviation of 0.5 for 
    each channel. It returns a DataLoader for the CIFAR-10 test set.

    Returns:
        None: The first return value is not used.
        torch.utils.data.DataLoader: DataLoader for the CIFAR-10 test set.
    """    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    testset = datasets.CIFAR10(root='../data/CIFAR', train=False,
                              download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False)
    return None, test_loader

def visualize_weights(network):
    """
    Visualizes the weights of the first convolutional layer of a neural network.
    
    This function extracts the weights from the first convolutional layer of the 
    network, reshapes them into a grid of 8x8, and then plots each filter as an 
    individual image. The images are saved in the './results' folder.

    Args:
        network (torch.nn.Module): The neural network model whose weights to visualize.
    """
    weights = network.conv1.weight.detach().cpu().numpy() 

    print("First layer filter weights shape:", weights.shape)

    # Create a 3x4 grid for visualization
    width, height = 8, 8
    fig, axes = plt.subplots(width, height, figsize=(8, 6))
    axes = axes.ravel()

    for i in range(width * height):
        axes[i].imshow(weights[i, 0], cmap="viridis") 
        axes[i].set_title(f"Filter {i}")
        axes[i].axis("off")

    plt.tight_layout()
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/weights_resnet.png')        
    plt.show()
    plt.close(fig)

def visualize_applied(network, test_loader):
    """
    Visualizes the application of the first convolutional layer filters on an image.
    
    This function takes the first image from the CIFAR-10 test loader, displays it,
    and applies each filter from the first convolutional layer of the network to 
    the image. Both the filter weights and the resulting filtered image are shown 
    in a grid. The images are saved in the './results' folder.

    Args:
        network (torch.nn.Module): The neural network model whose filters to visualize.
        test_loader (torch.utils.data.DataLoader): DataLoader for the CIFAR-10 test set.
    """

    # get the test loader and check 
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    image = images[0].numpy() # 32 x 32
    weights = network.conv1.weight.detach().cpu().numpy() 

    # show the image
    grey_image = images[0,0].numpy()
    plt.imshow(grey_image, cmap="gray")
    plt.axis('off')  # Turn off axis labels
    plt.show()

    # Create a 3x4 grid for visualization
    fig, axes = plt.subplots(9, 8, figsize=(12, 10))
    axes = axes.ravel()

    with torch.no_grad():
        for i in range(9 * 8):
            idx = i // 2
            filter_weights = weights[idx] # (3,7,7)

            if i % 2 == 0:
                filter_vis = np.mean(filter_weights, axis=0)  # (7,7)
                axes[i].imshow(filter_vis, cmap="Greys")
                axes[i].set_title(f"Weight: {idx}")
            else:
                # Apply filter to each channel and combine
                filtered_img = np.zeros((32, 32))
                for c in range(3):
                    filtered_img += cv2.filter2D(image[c], -1, filter_weights[c])
                filtered_img /= 3
                
                axes[i].imshow(filtered_img, cmap="Greys")
                axes[i].set_title(f"Applied: {idx}")
            axes[i].axis("off")

    plt.tight_layout()
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/applied.png')    
    plt.show()
    plt.close(fig)

def main():
    """
    The main function to load the data and visualize the filters and their application.

    This function loads the CIFAR-10 test dataset using `load_data()`, loads a 
    pretrained ResNet-50 model, and visualizes the first convolutional layer's 
    filters and their effect on the image using the `visualize_applied()` function.
    """
    _, test_loader = load_data()
    network = models.resnet50(pretrained=True)
    print(network)
    # visualize_weights(network)
    visualize_applied(network, test_loader)

if __name__ == "__main__":
    main()