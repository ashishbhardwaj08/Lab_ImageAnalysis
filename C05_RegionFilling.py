import cv2
import numpy as np

img = cv2.imread("Image.png", 0)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Copy image and create mask
h, w = binary.shape
mask = np.zeros((h+2, w+2), np.uint8)

# Flood fill from (0,0) background
floodfill = binary.copy()
cv2.floodFill(floodfill, mask, (0,0), 255)

# Invert floodfilled image
floodfill_inv = cv2.bitwise_not(floodfill)

# Combine with original to fill holes
filled = binary | floodfill_inv

cv2.imshow("Original", binary)
cv2.imshow("Region Filled", filled)
cv2.waitKey(0)
