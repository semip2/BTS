import cv2
import numpy as np

def process_vertical_segments(image_path):
    img = cv2.imread(image_path)
    img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    r, c, w = img_rotated.shape
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (c + 2000, 7))
    vertical = cv2.dilate(img_rotated, verticalStructure, (-1, -1))

    edges = cv2.Canny(vertical, 50, 150, apertureSize=3)

    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, minLineLength, maxLineGap)

    m = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            m.append(((x1, y1), (x2, y2)))

    sorted_m = sorted(m, key=lambda x: x[0][1])
    sorted_m.insert(0, ((0, 0), (c, 0)))

    for coords in sorted_m:
        cv2.line(img_rotated, coords[0], coords[1], (0, 0, 255), 3)

    segments = []
    for i in range(len(sorted_m)):
        if i != len(sorted_m) - 1:
            segments.append(img_rotated[sorted_m[i][0][1]:sorted_m[i+1][0][1], sorted_m[i][0][0]:sorted_m[i][1][0]])
        else:
            segments.append(img_rotated[sorted_m[-1][0][1]:r, sorted_m[-1][0][0]:sorted_m[-1][1][0]])

    return segments

def process_horizontal_segments(image_path):
    th2 = cv2.imread(image_path)
    r, c, w = th2.shape
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (c + 2000, 7))
    horizontal = cv2.dilate(th2, horizontalStructure, (-1, -1))

    edges = cv2.Canny(horizontal, 50, 150, apertureSize=3)

    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, minLineLength, maxLineGap)

    m = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            m.append(((x1, y1), (x2, y2)))

    sorted_m = sorted(m, key=lambda x: x[0][1])
    sorted_m.insert(0, ((0, 0), (c, 0)))

    for coords in sorted_m:
        cv2.line(th2, coords[0], coords[1], (0, 0, 255), 3)

    segments = []
    for i in range(len(sorted_m)):
        if i != len(sorted_m) - 1:
            segments.append(th2[sorted_m[i][0][1]:sorted_m[i+1][0][1], sorted_m[i][0][0]:sorted_m[i][1][0]])
        else:
            segments.append(th2[sorted_m[-1][0][1]:r, sorted_m[-1][0][0]:sorted_m[-1][1][0]])

    return segments

def main():
    image_path = 'segmentation/mask.jpg'
    vertical_segments = process_vertical_segments(image_path)
    horizontal_segments = process_horizontal_segments(image_path)

    # Display or save the results
    for i, segment in enumerate(vertical_segments):
        cv2.imwrite(f'vertical_part_{i}.jpg', segment)

    for i, segment in enumerate(horizontal_segments):
        cv2.imwrite(f'horizontal_part_{i}.jpg', segment)

if __name__ == "__main__":
    main()
