import torch
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from PIL import Image
from torch.nn.functional import softmax
import torchvision.transforms as transforms

def test_images():
    save_path = "final_model.pth"
    test_dir = "braille_test/"

    # Get all subdirectories in the test directory
    label_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]

    # Shuffle the label directories
    random.shuffle(label_dirs)

    # Sample 10 random label directories
    label_dirs = label_dirs[:10]

    # Load saved model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    model.load_state_dict(torch.load(save_path))
    model.eval()

    # Dictionary for all labels
    label_dict = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    _, axs = plt.subplots(2, 5, figsize=(20, 10))
    axs = axs.ravel()  # Flatten the 2D array of axes

    for i, label_dir in enumerate(label_dirs):
        # Get a random image from the label directory
        image_files = glob.glob(os.path.join(test_dir, label_dir, "*.jpg"))
        random_image_path = random.choice(image_files)

        # Preprocessing - should we change this?
        image = Image.open(random_image_path).convert("L")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to fit AlexNet input size
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.ToTensor(),           # Convert image to tensor
            transforms.Normalize(            # Normalize image
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        image = transform(image)

        # Perform inference
        with torch.no_grad():
            output = model(image.unsqueeze(0))
            probabilities = softmax(output, dim=1)[0]
            predicted_label_index = torch.argmax(probabilities).item()
            predicted_label = label_dict[predicted_label_index]
            actual_label = os.path.basename(label_dir)

        # Plot image and predicted label
        axs[i].imshow(image.permute(1, 2, 0))
        axs[i].set_title(f"Actual Label: {actual_label}\nPredicted Label: {predicted_label}")
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_images()
