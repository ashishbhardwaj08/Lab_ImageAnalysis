import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Image.jpg", 0)

# Global Threshold
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(thresh, cmap='gray')
plt.title("Global Threshold")
plt.axis("off")

plt.show()

