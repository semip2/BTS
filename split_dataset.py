import os
import shutil
import numpy as np

# Define paths
source_folder = 'sorted_files/'
train_output_folder = 'braille_train/'
val_output_folder = 'braille_val/'
test_output_folder = 'braille_test/'

# Define split ratios
train_ratio = 0.8
val_ratio = 0.1
# Test ratio would implicitly be 0.1 (100% - 80% - 10%)

# Creating output folders (if they don't exist)
os.makedirs(train_output_folder, exist_ok=True)
os.makedirs(val_output_folder, exist_ok=True)
os.makedirs(test_output_folder, exist_ok=True)

# For each class (Braille letter) in the dataset
for letter in os.listdir(source_folder):
    letter_folder = os.path.join(source_folder, letter)

    # Get files in the letter folder that match the format for images
    images = [img for img in os.listdir(letter_folder) if img.lower().endswith(('jpg'))]

    # Randomly shuffle the images
    np.random.shuffle(images)

    # Calculate split indices
    train_end = int(len(images) * train_ratio)
    val_end = int(len(images) * (train_ratio + val_ratio))

    # Split the images
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # Define the output paths for this letter
    train_letter_folder = os.path.join(train_output_folder, letter)
    val_letter_folder = os.path.join(val_output_folder, letter)
    test_letter_folder = os.path.join(test_output_folder, letter)

    # Create corresponding letter subdirectories
    os.makedirs(train_letter_folder, exist_ok=True)
    os.makedirs(val_letter_folder, exist_ok=True)
    os.makedirs(test_letter_folder, exist_ok=True)

    # Move the files to the corresponding output folders
    for image in train_images:
        shutil.move(os.path.join(letter_folder, image), train_letter_folder)
      
    for image in val_images:
        shutil.move(os.path.join(letter_folder, image), val_letter_folder)

    for image in test_images:
        shutil.move(os.path.join(letter_folder, image), test_letter_folder)