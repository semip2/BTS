from __future__ import print_function
import cv2
import numpy as np
import math
import string
import sys



# f=open("C:\Users\Mahe\Desktop\MUSOC.txt","w+")

img=cv2.imread('part0.jpg')
rows,cols,w = img.shape

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, threshold = cv2.threshold(gray_image,200,255,cv2.THRESH_BINARY)


#dilating
kernel = np.ones((100,10),np.uint8)
dilation = cv2.dilate(threshold,kernel,iterations = 1)

cv2.imwrite(r'dilatedpart.jpg', dilation)
k=cv2.imread('dilatedpart.jpg')

#defining the edges
edges = cv2.Canny(k,50,150,apertureSize = 3)
cv2.imwrite('cannyEdge1.jpg',edges)

m=[]
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
                m.append(((x1,y1),(x2,y2)))


#sorting list m as per x coordinate in the tuple
sorted_m=sorted(m, key=lambda x: x[0][0])
#     print(sorted_m)

#     cv2.line(img,sorted_m[0][0],sorted_m[0][1],(0,0,255),2)
#     cv2.line(img,sorted_m[-1][0],sorted_m[-1][1],(0,0,255),2)
#     cv2.imwrite(r'hough1.jpg',img)

for line in range(len(sorted_m)):
        cur_x = sorted_m[line][0][0]
        if(line + 1 > len(sorted_m) - 1) or (line + 2 > len(sorted_m) - 2):
                break
        next_x = sorted_m[line + 1][0][0]
        far_x = sorted_m[line + 2][0][0]
        if next_x - cur_x < 30:
                sorted_m.pop(line + 1)
        if far_x - cur_x < 30:
                sorted_m.pop(line + 2)

sorted_m.append(((367, 48), (367, 0)))
sorted_m.append(((660, 48), (660, 0)))
sorted_m.append(((885, 48), (885, 0)))
sorted_m.append(((920, 48), (920, 0)))
sorted_m.pop(5)
sorted_m.pop(17)
sorted_m = sorted(sorted_m, key=lambda x: x[0][0])



#     drawing lines
for i in range(len(sorted_m)):
        cv2.line(img,sorted_m[i][0],sorted_m[i][1],(0,0,255),2)
cv2.imwrite(r'houghfinal.jpg',img)

#     num_boxes = 24
#     box_width = sorted_m[-1][0][0] - sorted_m[0][0][0]
#     start = sorted_m[0]
#     print(start)
#     distance = int(box_width / num_boxes)
#     for i in range(1, num_boxes):
#         cv2.line(img, (start[0][0] + distance*i, start[0][1]), (start[1][0] + distance*i, start[1][1]), (0,0,255), 2)
#     cv2.imwrite(r'hough_divided.jpg',img)