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


def main(argv):
    cuda_is_avail = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"cuda available?:  {cuda_is_avail}")
    print(f"Using device: {device}")

    print(torch.__version__)
    print(torch.version.cuda)


if __name__ == "__main__":
    main(sys.argv)
