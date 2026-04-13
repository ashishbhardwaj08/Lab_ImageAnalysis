
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = cv2.imread("Image2.jpg", 0)

# Create Gabor kernel
kernel = cv2.getGaborKernel((21, 21),
                            5.0,
                            np.pi/4,
                            10.0,
                            0.5,
                            0,
                            ktype=cv2.CV_32F)

filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)

# Flatten and cluster
features = filtered.reshape(-1, 1)

kmeans = KMeans(n_clusters=2)
labels = kmeans.fit_predict(features)

segmented = labels.reshape(img.shape)

plt.imshow(segmented, cmap='gray')
plt.title("Texture Segmentation - Gabor")
plt.show()
