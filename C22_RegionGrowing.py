import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Image3.png", 0)

# Seed point
seed = (100, 100)
threshold = 10

segmented = np.zeros_like(img)
seed_value = img[seed]

rows, cols = img.shape

stack = [seed]

while stack:
    x, y = stack.pop()

    if segmented[x, y] == 0:
        if abs(int(img[x, y]) - int(seed_value)) < threshold:
            segmented[x, y] = 255

            # Add neighbors
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < rows and 0 <= ny < cols:
                        stack.append((nx, ny))

plt.imshow(segmented, cmap='gray')
plt.title("Region Growing")
plt.axis("off")
plt.show()
