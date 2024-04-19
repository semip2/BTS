import cv2
import numpy as np
import matplotlib.pyplot as plt

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def merge_close_points(points, threshold):
    merged_points = []
    merged_indices = set()  # To keep track of merged points

    for i, point1 in enumerate(points):
        if i in merged_indices:
            continue

        merged_point = point1
        for j, point2 in enumerate(points[i + 1:], start=i + 1):
            if j in merged_indices:
                continue

            if distance(point1, point2) < threshold:
                merged_point = ((merged_point[0] + point2[0]) / 2, (merged_point[1] + point2[1]) / 2)
                merged_indices.add(j)

        merged_points.append(merged_point)

    return merged_points


# Load your image
image = cv2.imread('samples/alyssa2.png', cv2.IMREAD_GRAYSCALE)  # Load as grayscale

# Perform Canny edge detection
canny_edges = cv2.Canny(image, threshold1=50, threshold2=60)  # Adjust thresholds as needed
# plt.imshow(canny_edges)
# plt.show()

num_labels, labels, centroid_stats, centroids = cv2.connectedComponentsWithStats(canny_edges, connectivity=8)
sorted_centroids = sorted(centroids[1:], key=lambda x: (x[1], x[0]))
print(len(sorted_centroids)) 
merged_points = merge_close_points(sorted_centroids, threshold=10.0)
print(len(merged_points)) 

white_canvas = np.ones_like(image) * 255

lines_canvas = white_canvas.copy()
width, height = image.shape
for centroid in merged_points:
    x, y = int(round(centroid[0])), int(round(centroid[1]))
    cv2.circle(image, (x, y), radius=1, color=(255,0,0), thickness=-1)
    
    cv2.line(lines_canvas, (x, 0), (x, height*100000), color=(0,0,0), thickness=1)
    cv2.line(lines_canvas, (0, y), (width*100000, y), color=(0,0,0), thickness=1)

plt.imshow(image)
plt.title('Braille con Centroides Marcados')
plt.axis('off')
plt.show()

plt.imshow(lines_canvas)
plt.title('Braille con Líneas de Cuadrícula')
plt.axis('off')
plt.show()
