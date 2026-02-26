import cv2
import numpy as np

img = cv2.imread("Image.jpg", 0)

kernel = np.ones((5,5), np.uint8)

gray_dilate = cv2.dilate(img, kernel)

cv2.imshow("Original", img)


cv2.imshow("Grayscale Dilation", gray_dilate)
cv2.waitKey(0)


gray_erode = cv2.erode(img, kernel)

cv2.imshow("Grayscale Erosion", gray_erode)
cv2.waitKey(0)

gray_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

cv2.imshow("Grayscale Opening", gray_open)
cv2.waitKey(0)


gray_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Grayscale Closing", gray_close)
cv2.waitKey(0)
