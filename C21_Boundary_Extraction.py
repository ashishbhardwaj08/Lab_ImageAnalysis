import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("Image5.png")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 100, 200)

contours, _ = cv2.findContours(edges, 
                               cv2.RETR_EXTERNAL, 
                               cv2.CHAIN_APPROX_SIMPLE)

boundary_img = img.copy()
cv2.drawContours(boundary_img, contours, -1, (0,255,0), 2)

cv2.imshow("Boundary analysis", boundary_img)


plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(edges, cmap='gray')
plt.title("Edges")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(boundary_img, cv2.COLOR_BGR2RGB))
plt.title("Boundaries")
plt.axis("off")

plt.show()
