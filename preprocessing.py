import cv2
import argparse
import numpy as np

img= cv2.imread('samples/processed.jpg',0)
    
# # Apply Gaussian blur to reduce noise
# # Possibility of Adaptive Blur
# blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# # Apply adaptive thresholding to binarize the inverted image
# binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)

# # Apply morphological operations to enhance transitions
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

img = 255 - img

ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
blur = cv2.blur(thresh,(5,5))

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(blur,kernel,iterations = 1)
ret, thresh2 = cv2.threshold(erosion, 12, 255, cv2.THRESH_BINARY)

kernel = np.ones((3,2),np.uint8)
mask = cv2.dilate(thresh2,kernel,iterations = 1)

rows,cols=mask.shape
cv2.imwrite(r'segmentation/mask.jpg', mask)