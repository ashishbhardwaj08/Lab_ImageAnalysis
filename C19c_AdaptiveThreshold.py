import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Image.jpg", 0)

# Global Threshold
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)


adaptive = cv2.adaptiveThreshold(
    img, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    11, 2)

cv2.imshow("Adaptive Threshold", adaptive)
cv2.waitKey(0)