import torch
import glob
import os
from PIL import Image
from torch.nn.functional import softmax
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
from natsort import natsorted

# Custom transform for inverting the colors
class InvertColors:
    def __call__(self, img):
        return 1 - img  # Subtract image tensor from 1 to invert colors

def test_images():
    save_path = "final_model.pth"
    base_dir = "example/segments/"
    line_dirs = natsorted(glob.glob(os.path.join(base_dir, "line*")))

    # Load saved model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    model.load_state_dict(torch.load(save_path))
    model.eval()

    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to fit AlexNet input size
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.ToTensor(),           # Convert image to tensor
        InvertColors(),
        transforms.Normalize(            # Normalize image
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    for line_dir in line_dirs:
        # Iterate through each file in the directory recursively
        image_files = natsorted(glob.glob(os.path.join(line_dir, "*.jpg")))

        if not image_files:  # Check if the list is empty
            print(f"No images found in directory: {line_dir}")
            continue

        num_images = len(image_files)
        cols = min(num_images, 10)  # Limit to 10 images per row
        rows = math.ceil(num_images / cols)

        plt.figure(figsize=(24, 4 * rows))
        for i, image_path in enumerate(image_files):
            # Load and preprocess the image
            image = Image.open(image_path).convert("L")
            image = transform(image)

            # Perform inference
            with torch.no_grad():
                output = model(image.unsqueeze(0))
                probabilities = softmax(output, dim=1)[0]
                predicted_label_index = torch.argmax(probabilities).item()
                predicted_label = chr(predicted_label_index + 65)  # Assuming labels are A-Z

            # Plot results
            plt.subplot(rows, cols, i + 1)
            plt.imshow(image.permute(1, 2, 0), cmap='gray')
            plt.title(f"File: {os.path.basename(image_path)}\nPredicted: {predicted_label}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_images()
