import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Image.jpg", 0)

# Global Threshold
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)


# Define window range
T1 = 100
T2 = 180

window = np.zeros_like(img)

window[(img >= T1) & (img <= T2)] = 255

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(window, cmap='gray')
plt.title("Window Slicing")
plt.axis("off")

plt.show()
