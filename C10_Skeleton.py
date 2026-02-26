# -- NOT WORKING CORRECTLY--#


#Using distance transform
import cv2
import numpy as np

img = cv2.imread("Image5.png", 0)

#Invert
# original = cv2.imread("Image.png", 0)
# img = cv2.bitwise_not(original)





_, binary = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

# Distance Transform
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

# Normalize for display
dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
dist_norm = np.uint8(dist_norm)

# Threshold distance map to get skeleton
_, skeleton = cv2.threshold(dist_norm, 100, 255, cv2.THRESH_BINARY)

cv2.imshow("Original", binary)
cv2.imshow("Distance Map", dist_norm)
cv2.imshow("Skeleton", skeleton)
cv2.waitKey(0)
