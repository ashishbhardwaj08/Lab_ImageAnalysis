import cv2
import numpy as np

# Load image
img = cv2.imread("Image.jpg")

# Get image shape
h, w, c = img.shape

# Reshape image to a 2D array of pixels
pixels = img.reshape(-1, 3)

# Sort pixels by B, then G, then R (ascending)
pixels_sorted = pixels[np.lexsort((pixels[:,2], pixels[:,1], pixels[:,0]))]

# Reshape back to image shape
sorted_img = pixels_sorted.reshape(h, w, 3)

# Show images
cv2.imshow("Original Image", img)
cv2.imshow("Sorted Image", sorted_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
# cv2.imwrite("sorted_image.jpg", sorted_img)
