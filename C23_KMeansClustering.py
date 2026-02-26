import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Image2.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Reshape image
pixel_values = img_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Define criteria
criteria = (cv2.TERM_CRITERIA_EPS + 
            cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

K = 3
_, labels, centers = cv2.kmeans(pixel_values, 
                                 K, None, 
                                 criteria, 
                                 10, 
                                 cv2.KMEANS_RANDOM_CENTERS)

# Convert back to image
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape(img_rgb.shape)

plt.figure(figsize=(16,8))

plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(segmented_image)
plt.title("K-Means Segmentation")
plt.axis("off")

plt.show()



# cv2.imshow("K-Means Segmentation", segmented_image)
# cv2.waitKey(0)
