"""
Name: [Your Name]
Project 5: Recognition using Deep Networks - Extended Greek Letters
Date: Mar 27, 2025
This script prepares the directory structure for the extended Greek letters dataset.
"""

import os
import shutil
from pathlib import Path

# Define Greek letters to include
GREEK_LETTERS = ['alpha', 'beta', 'gamma',
                 'delta', 'epsilon', 'zeta', 'eta', 'theta']

# Define paths
DATA_DIR = Path('data')
GREEK_TRAIN_DIR = DATA_DIR / 'greek_extended'
GREEK_CUSTOM_DIR = DATA_DIR / 'greek_custom_extended'


def create_directory_structure():
    """
    Creates the directory structure for the extended Greek letters dataset.
    """
    print("Creating directory structure for extended Greek letters dataset...")

    # Create main directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Create or clear training directory
    if os.path.exists(GREEK_TRAIN_DIR):
        print(
            f"Directory {GREEK_TRAIN_DIR} already exists. Do you want to clear it? (y/n)")
        response = input().lower()
        if response == 'y':
            shutil.rmtree(GREEK_TRAIN_DIR)
            os.makedirs(GREEK_TRAIN_DIR)
    else:
        os.makedirs(GREEK_TRAIN_DIR)

    # Create subdirectories for each Greek letter
    for letter in GREEK_LETTERS:
        os.makedirs(GREEK_TRAIN_DIR / letter, exist_ok=True)
        print(f"Created directory: {GREEK_TRAIN_DIR / letter}")

    # Create or clear custom test directory
    if os.path.exists(GREEK_CUSTOM_DIR):
        print(
            f"Directory {GREEK_CUSTOM_DIR} already exists. Do you want to clear it? (y/n)")
        response = input().lower()
        if response == 'y':
            shutil.rmtree(GREEK_CUSTOM_DIR)
            os.makedirs(GREEK_CUSTOM_DIR)
    else:
        os.makedirs(GREEK_CUSTOM_DIR)

    # Create subdirectories for each Greek letter in the custom directory
    for letter in GREEK_LETTERS:
        os.makedirs(GREEK_CUSTOM_DIR / letter, exist_ok=True)
        print(f"Created directory: {GREEK_CUSTOM_DIR / letter}")

    print("\nDirectory structure created successfully!")
    print("\nAdd your training images to the corresponding subdirectories in:")
    print(f"{GREEK_TRAIN_DIR}")
    print("\nAdd your custom test images to the corresponding subdirectories in:")
    print(f"{GREEK_CUSTOM_DIR}")


def reuse_existing_greek_data():
    """
    Copies existing alpha, beta, and gamma data from the original Greek dataset.
    """
    original_data_dir = DATA_DIR / 'greek_train'

    if not os.path.exists(original_data_dir):
        print(f"Original Greek dataset not found at {original_data_dir}")
        return False

    print("Copying existing Greek data from the original dataset...")

    # Copy alpha, beta, and gamma data
    for letter in ['alpha', 'beta', 'gamma']:
        src_dir = original_data_dir / letter
        dst_dir = GREEK_TRAIN_DIR / letter

        if not os.path.exists(src_dir):
            print(f"Warning: Source directory {src_dir} not found")
            continue

        # Copy all files from source to destination
        for file in os.listdir(src_dir):
            src_file = src_dir / file
            dst_file = dst_dir / file

            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
                print(f"Copied {src_file} to {dst_file}")

    print("Existing Greek data copied successfully!")
    return True


def main():
    # Create the directory structure
    create_directory_structure()

    # Ask if the user wants to copy existing data
    print("\nDo you want to copy existing alpha, beta, and gamma data from the original dataset? (y/n)")
    response = input().lower()

    if response == 'y':
        success = reuse_existing_greek_data()
        if success:
            print("\nNext steps:")
            print("1. Add images for the additional Greek letters (delta, epsilon, zeta, etc.) to their respective folders")
            print("2. Run the extended_greek_network.py script to train the network")
        else:
            print(
                "\nCould not copy existing data. You'll need to add all Greek letter images manually.")
    else:
        print("\nYou'll need to add all Greek letter images manually to their respective folders.")


if __name__ == "__main__":
    main()
