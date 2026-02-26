import cv2
import numpy as np

img = cv2.imread("Image5.png", 0)
_, binary = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

kernel = np.ones((3,3), np.uint8)

eroded = cv2.erode(binary, kernel, iterations=1)

boundary = cv2.subtract(binary, eroded)

cv2.imshow("Original", binary)
cv2.imshow("Eroded", eroded)
cv2.imshow("Boundary", boundary)
cv2.waitKey(0)
