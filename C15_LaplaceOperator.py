import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
img = cv2.imread("Image5.png", 0)

# -------- Using OpenCV Built-in Laplacian --------
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Convert to uint8
laplacian_abs = np.uint8(np.absolute(laplacian))


# -------- Using Manual 4-neighbour Kernel --------
kernel_4 = np.array([[0,-1,0],
                     [-1,4,-1],
                     [0,-1,0]])

lap_4 = cv2.filter2D(img, cv2.CV_64F, kernel_4)
lap_4 = np.uint8(np.absolute(lap_4))


# -------- Using Manual 8-neighbour Kernel --------
kernel_8 = np.array([[-1,-1,-1],
                     [-1,8,-1],
                     [-1,-1,-1]])

lap_8 = cv2.filter2D(img, cv2.CV_64F, kernel_8)
lap_8 = np.uint8(np.absolute(lap_8))


# -------- Display --------
plt.figure(figsize=(12,5))

plt.subplot(1,4,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(laplacian_abs, cmap='gray')
plt.title("OpenCV Laplacian")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(lap_4, cmap='gray')
plt.title("4-Neighbour")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(lap_8, cmap='gray')
plt.title("8-Neighbour")
plt.axis("off")

plt.tight_layout()
plt.show()
