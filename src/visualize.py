import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from mnist_network import MyNetwork, load_data, test_network

MODEL_PATH = "../models/mnist_network.pth"

def visualize_weights(network):
    weights = network.conv1.weight.detach().cpu().numpy() 

    print("First layer filter weights shape:", weights.shape)

    # Create a 3x4 grid for visualization
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    axes = axes.ravel()

    for i in range(12):
        if i < 10: # for the two last, only erase the axis
            axes[i].imshow(weights[i, 0], cmap="viridis") 
            axes[i].set_title(f"Filter {i}")
        axes[i].axis("off")

    plt.tight_layout()
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/weights.png')        
    plt.show()
    plt.close(fig)

def visualize_applied(network, test_loader):

    # get the test loader and check 
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    image = images[0,0].numpy()

    weights = network.conv1.weight.detach().cpu().numpy() 

    # Create a 3x4 grid for visualization
    fig, axes = plt.subplots(5, 4, figsize=(8, 6))
    axes = axes.ravel()

    with torch.no_grad():
        for i in range(5 * 4):
            idx = i // 2
            if i % 2 == 0:
                axes[i].imshow(weights[idx, 0], cmap="Greys")
                axes[i].set_title(f"Weight: {idx}")
            else:
                filtered_img = cv2.filter2D(image, -1, weights[idx, 0])
                axes[i].imshow(filtered_img, cmap="Greys")
                axes[i].set_title(f"Applied: {idx}")
            axes[i].axis("off")

    plt.tight_layout()
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/applied.png')    
    plt.show()
    plt.close(fig)

def main():
    _, test_loader = load_data()
    network = MyNetwork()
    network.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    # test_network(network, test_loader, False)
    # visualize_weights(network)
    visualize_applied(network, test_loader)

if __name__ == "__main__":
    main()