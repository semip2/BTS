import cv2
import numpy as np

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

def vertical_segmentation(binary_image):
    # Sum the pixel values vertically to find transitions between lines
    vertical_projection = cv2.reduce(binary_image, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    
    # Threshold the vertical projection to find transitions
    _, _, _, max_loc = cv2.minMaxLoc(vertical_projection)
    _, binary_threshold = cv2.threshold(vertical_projection, max_loc[1] * 0.5, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to enhance transitions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_threshold = cv2.morphologyEx(binary_threshold, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of transitions
    contours, _ = cv2.findContours(binary_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get vertical coordinates of transitions
    transitions = [cv2.boundingRect(contour)[1] for contour in contours]
    transitions.sort()
    
    return transitions


def horizontal_segmentation(binary_image, transitions):
    segmented_characters = []
    
    for i in range(len(transitions) - 1):
        # Crop the image to a single line
        line_image = binary_image[transitions[i]:transitions[i + 1], :]
        
        # Sum the pixel values horizontally to find transitions between characters
        horizontal_projection = cv2.reduce(line_image, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
        
        # Threshold the horizontal projection to find transitions
        _, _, _, max_loc = cv2.minMaxLoc(horizontal_projection)
        _, binary_threshold = cv2.threshold(horizontal_projection, max_loc[1] * 0.5, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to enhance transitions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_threshold = cv2.morphologyEx(binary_threshold, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of transitions
        contours, _ = cv2.findContours(binary_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get horizontal coordinates of transitions
        for contour in contours:
            x, _, w, _ = cv2.boundingRect(contour)
            segmented_characters.append((x, transitions[i], w, transitions[i + 1] - transitions[i]))
    
    return segmented_characters


def display_segmented_characters(image, segmented_characters):
    for idx, (x, y, w, h) in enumerate(segmented_characters):
        character_region = image[y:y+h, x:x+w]
        cv2.imshow(f'Character {idx}', character_region)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    # Path to the input image containing Braille text
    image_path = 'braille_text_image.jpg'
    
    # Preprocess the image
    binary_image = preprocess_image(image_path)
    
    # Perform vertical segmentation
    transitions = vertical_segmentation(binary_image)
    
    # Perform horizontal segmentation
    segmented_characters = horizontal_segmentation(binary_image, transitions)
    
    # Display segmented characters
    display_segmented_characters(binary_image, segmented_characters)

if __name__ == "__main__":
    main()