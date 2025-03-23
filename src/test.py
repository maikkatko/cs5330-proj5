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
from mnist_network import MyNetwork, load_data, test_network

MODEL_PATH = "./models/mnist_network.pth"

def scale_image(path: str):
    '''
    Scale the image into 28x28 greyscale from the 
    '''
    pass 

def load_data(path: str):
    '''
    Load the data as custom_data.
    '''
    pass

def process_handwritten_digits(network, custom_data):
    """
    Test the network with the custom data.
    """
    pass

def main():
    _, test_loader = load_data()
    network = MyNetwork()
    network.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    test_network(network, test_loader)

if __name__ == "__main__":
    main()