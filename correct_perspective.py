import cv2, numpy as np
import sys
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    # Read the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur to reduce noise
    # Possibility of Adaptive Blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply adaptive thresholding to binarize the inverted image
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)

    # Apply morphological operations to enhance transitions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    return binary_image


# Load your image
image = preprocess_image('braille_slanted.jpg') 
canny_edges = cv2.Canny(image, threshold1=50, threshold2=60)  # Adjust thresholds as needed
plt.imshow(canny_edges) 
plt.show() 
contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest rectangle among the contours
largest_area = 0
largest_contour = None
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = approx

# Reshape the largest contour to a standard format
largest_contour = np.array(largest_contour).reshape(-1, 2)

# Define the desired shape (a rectangle)
desired_shape = np.array([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]], dtype=np.float32)

# Compute the perspective transform
M = cv2.getPerspectiveTransform(largest_contour.astype(np.float32), desired_shape)

# Apply perspective transform
warped_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

plt.imshow(warped_image) 
plt.show() 

