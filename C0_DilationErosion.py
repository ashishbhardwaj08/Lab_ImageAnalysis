import cv2
import numpy as np
# Load image
img = cv2.imread("Image5.png", cv2.IMREAD_GRAYSCALE)
# Convert to binary
_, binary = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
# Structuring element (kernel)
kernel = np.ones((5,5), np.uint8)
# Dilation
dilation = cv2.dilate(binary, kernel, iterations=5)
# Erosion
erosion = cv2.erode(binary, kernel, iterations=5   )
# Show results
cv2.imshow("Original", binary)
cv2.imshow("Dilation", dilation)
cv2.imshow("Erosion", erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()