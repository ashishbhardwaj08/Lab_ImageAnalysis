import cv2
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("Image.jpg", 0)

# Apply Canny
edges = cv2.Canny(img, 100, 200)

# plt.figure(figsize=(8,4))

# plt.subplot(1,2,1)
# plt.imshow(img, cmap='gray')
# plt.title("Original")
# plt.axis("off")

# plt.subplot(1,2,2)
# plt.imshow(edges, cmap='gray')
# plt.title("Canny Edge Detection")
# plt.axis("off")

# plt.show()


cv2.imshow("Canny Edge Detection",edges)
cv2.waitKey(0)
