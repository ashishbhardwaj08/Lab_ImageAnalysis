import cv2
import numpy as np


img = cv2.imread("Image5.png", 0)

_, binary = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

# Connected components
num_labels, labels = cv2.connectedComponents(binary, connectivity=8)

print("Number of connected components:", num_labels)

# Color each component
output = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)

for i in range(1, num_labels):
    mask = labels == i
    output[mask] = np.random.randint(0,255,3)

cv2.imshow("Connected Regions", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
