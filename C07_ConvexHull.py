import cv2
import numpy as np

img = cv2.imread("Image.png", 0)


_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

hull_img = np.zeros_like(binary)

for cnt in contours:
    hull = cv2.convexHull(cnt)
    cv2.drawContours(hull_img, [hull], -1, 255, -1)

cv2.imshow("Original", binary)
cv2.imshow("Convex Hull", hull_img)
cv2.waitKey(0)
