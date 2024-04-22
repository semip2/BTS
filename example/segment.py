import os 
import shutil
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def preprocess(img): 
    inverted = 255 - img
    _, thresh1 = cv2.threshold(inverted, 180, 255, cv2.THRESH_BINARY)
    denoised = cv2.bilateralFilter(thresh1, 9, 75, 75) 
    _, thresh2 = cv2.threshold(denoised, 12, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh2, np.ones((3,2),np.uint8), iterations = 1)
    print("Finished preprocess") 
    return dilated 


def correct_perspective(src): 
    H, W = src.shape[:2] 
    frame = cv2.resize(src, (W, H)) 

    # Corners - Hard coded 
    tl = (13 - 11, 45 - 11) 
    bl = (14 - 11, 608 + 11) 
    tr = (736 + 11, 31 - 11) 
    br = (747 + 11, 605 + 11) 

    pts_src = np.float32([tl, bl, tr, br])
    pts_dst = np.float32([[0, 0], [0, H - 1], [W - 1, 0], [W - 1, H - 1]]) 

    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst) 
    transformed = cv2.warpPerspective(frame, matrix, (W, H))
    print("Finished correct perspective") 
    return transformed 


def horizontal_segment(img): 
    horizontal_img = img.copy() 

    # Hough lines 
    H, W = horizontal_img.shape[:2] 
    structure = cv2.getStructuringElement(cv2.MORPH_RECT, (W + 2000, 7))
    horizontal = cv2.dilate(horizontal_img, structure, (-1, -1))
    edges = cv2.Canny(horizontal, 50, 150, apertureSize = 3)

    # Get endpoints  
    y = []
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, minLineLength=100, maxLineGap=200)
    for i in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            y.append(((x1, y1), (x2, y2)))

    y.append(((0, 0), (W - 1, 0)))
    y.append(((0, H - 1), (W - 1, H - 1))) 
    sorted_y = sorted(y) 

    # Select segments to keep 
    segments = [] 
    for i in range(len(sorted_y) - 1): 
        y1 = sorted_y[i]
        y2 = sorted_y[i + 1] 

        segment = horizontal_img[y1[0][1]:y2[0][1], y1[0][0]:y1[1][0]]
        total = (y2[0][1] - y1[0][1]) * (y1[1][0] - y1[0][0])
        count = np.floor(np.sum(segment) / 255) 

        if (count/total > 0.005): 
            segments.append((y1[1][1], y2[1][1])) 

    # Merge broken segments  
    merged_segments = [] 
    heights = [s[1] - s[0] for s in segments]
    threshold = 0.8 * np.median(heights) 
    i = 0 
    while i < len(heights): 
        if i == len(heights) - 1 or heights[i] > threshold: 
            merged_segments.append(segments[i]) 
            i += 1 
        elif i == len(heights) - 2 or heights[i] + heights[i + 1] > threshold: 
            merged_segments.append((segments[i][0], segments[i + 1][1])) 
            i += 2 
        elif i == len(heights) - 3 or heights[i] + heights[i + 1] + heights[i + 2] > threshold: 
            merged_segments.append((segments[i][0], segments[i + 2][1])) 
            i += 3 

    for i in range(0, len(merged_segments)): 
        cv2.line(horizontal_img, (0, merged_segments[i][0]), (W - 1, merged_segments[i][0]), (0, 0, 255), 3) 
        cv2.line(horizontal_img, (0, merged_segments[i][1]), (W - 1, merged_segments[i][1]), (255, 0, 0), 3) 
    print("Finished horizontal segmentation") 
    return merged_segments 


def vertical_segment(img): 
    vertical_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # Use Hough lines 
    H, W = vertical_img.shape[:2] 
    structure = cv2.getStructuringElement(cv2.MORPH_RECT, (W + 2000, 7))
    horizontal = cv2.dilate(vertical_img, structure, (-1, -1))
    edges = cv2.Canny(horizontal, 50, 150, apertureSize = 3)

    # Get endpoints  
    y = []
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, minLineLength=100, maxLineGap=200)
    for i in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            y.append(((x1, y1), (x2, y2)))

    y.append(((0, 0), (W, 0)))
    y.append(((0, H - 1), (W - 1, H - 1))) 
    sorted_y = sorted(y) 

    # Keep segments with Braille 
    segments = [] 
    for i in range(len(sorted_y) - 1): 
        y1 = sorted_y[i]
        y2 = sorted_y[i + 1] 

        segment = vertical_img[y1[0][1]:y2[0][1], y1[0][0]:y1[1][0]]
        total = (y2[0][1] - y1[0][1]) * (y1[1][0] - y1[0][0])
        count = np.floor(np.sum(segment) / 255) 

        if (count/total > 0.01): 
            segments.append((sorted_y[i][1][1], sorted_y[i + 1][1][1])) 

    # Merge broken segments  
    merged_segments = [] 
    heights = [s[1] - s[0] for s in segments]
    threshold = 0.8 * np.median(heights) 
    i = 0 

    while i < len(heights): 
        if i == len(heights) - 1 or heights[i] > threshold: 
            merged_segments.append(segments[i]) 
            i += 1 
        elif i == len(heights) - 2 or heights[i] + heights[i + 1] > threshold: 
            merged_segments.append((segments[i][0], segments[i + 1][1])) 
            i += 2 

    # for i in range(0, len(merged_segments)): 
    #     cv2.line(vertical_img, (0, merged_segments[i][0]), (W - 1, merged_segments[i][0]), (0, 0, 255), 3) 
    #     cv2.line(vertical_img, (0, merged_segments[i][1]), (W - 1, merged_segments[i][1]), (255, 0, 0), 3) 
    # vertical_img = cv2.rotate(vertical_img, cv2.ROTATE_90_COUNTERCLOCKWISE) 
    print("Finished vertical segmentation") 
    return merged_segments 


def segment(img, x_pairs, y_pairs): 
    print("Starting segmentation") 
    line_num = 0

    # Create outer directory if it doesn't exist
    outer_directory = "segments"
    if not os.path.exists(outer_directory):
        os.makedirs(outer_directory)
    else:
        # Delete all the subdirectories in the outer directory
        for item in os.listdir(outer_directory):
            item_path = os.path.join(outer_directory, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)

    for y_start, y_end in y_pairs: 
        char_num = 0 
        empty_count = 0 

        for x_start, x_end in x_pairs: 
            segment = img[y_start:y_end, x_start:x_end]
            total = (y_end - y_start) * (x_end - x_start) 
            count = np.floor(np.sum(segment) / 255) 

            if (count/total < 0.01): 
                empty_count += 1 
                if empty_count > 1: 
                    break 
            else: 
                empty_count = 0 

            directory = f"segments/line{line_num}"
            if not os.path.exists(directory):
                os.makedirs(directory)

            cv2.imwrite(f"{directory}/char{char_num}.jpg", segment)
            char_num += 1 

        line_num += 1 

    print("Finished segmentation") 
    return 


# Load image 
img = cv2.imread('poem.png') 

# Preprocess
preprocessed = preprocess(img) 
cv2.imwrite('preprocessed.jpg', preprocessed) 

# Transform 
transformed = correct_perspective(preprocessed) 
cv2.imwrite('transformed.jpg', transformed) 

# Draw lines 
x_pairs = vertical_segment(transformed) 
y_pairs = horizontal_segment(transformed) 

# Visualize 
segmented = transformed.copy() 
H, W = segmented.shape[:2] 

for i in range(len(x_pairs)): 
    cv2.line(segmented, (x_pairs[i][0], 0), (x_pairs[i][0], H - 1), (0, 0, 255), 3) 
    cv2.line(segmented, (x_pairs[i][1], 0), (x_pairs[i][1], H - 1), (255, 0, 0), 3) 

for i in range(0, len(y_pairs)): 
    cv2.line(segmented, (0, y_pairs[i][0]), (W - 1, y_pairs[i][0]), (0, 0, 255), 3) 
    cv2.line(segmented, (0, y_pairs[i][1]), (W - 1, y_pairs[i][1]), (255, 0, 0), 3) 

cv2.imwrite('segmented.jpg', segmented) 

# Segment 
segment(transformed, x_pairs, y_pairs) 