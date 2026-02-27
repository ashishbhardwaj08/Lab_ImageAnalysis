import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Image.jpg", 0)

# Step 1: Gaussian Blur
blur = cv2.GaussianBlur(img, (5,5), 1)

# Step 2: Laplacian
lap = cv2.Laplacian(blur, cv2.CV_64F)

# Step 3: Zero Crossing Detection
zero_cross = np.zeros(lap.shape, dtype=np.uint8)

rows, cols = lap.shape

for i in range(1, rows-1):
    for j in range(1, cols-1):
        patch = lap[i-1:i+2, j-1:j+2]
        p = lap[i,j]

        if p > 0:
            if np.any(patch < 0):
                zero_cross[i,j] = 255
        elif p < 0:
            if np.any(patch > 0):
                zero_cross[i,j] = 255

# Display results
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(np.absolute(lap), cmap='gray')
plt.title("Laplacian")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(zero_cross, cmap='gray')
plt.title("Zero Crossing")
plt.axis("off")

plt.tight_layout()
plt.show()
