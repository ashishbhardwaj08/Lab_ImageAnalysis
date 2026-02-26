import cv2
import numpy as np

img = cv2.imread("Image.png", 0)





_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

num_labels, labels = cv2.connectedComponents(binary)

print("Number of objects:", num_labels-1)  # excluding background

# Color each component
colored = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)

for label in range(1, num_labels):
    colored[labels == label] = np.random.randint(0,255, size=3)

cv2.imshow("Connected Components", colored)
cv2.waitKey(0)
