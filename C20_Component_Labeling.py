import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
img = cv2.imread("Image5.png", 0)

# Threshold to get binary image
_, binary = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

# Connected Component Analysis
num_labels, labels = cv2.connectedComponents(binary)

print("Number of components:", num_labels - 1)

# Convert labels to color image for visualization
label_hue = np.uint8(179 * labels / np.max(labels))
blank = 255 * np.ones_like(label_hue)
colored = cv2.merge([label_hue, blank, blank])
colored = cv2.cvtColor(colored, cv2.COLOR_HSV2BGR)
colored[label_hue == 0] = 0

cv2.imshow("Components", colored)

# Display
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(binary, cmap='gray')
plt.title("Binary Image")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(colored)
plt.title("Labeled Components")
plt.axis("off")

plt.show()
