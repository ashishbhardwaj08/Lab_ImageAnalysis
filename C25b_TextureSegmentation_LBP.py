

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

img = cv2.imread("Image4.png", 0)

radius = 1
n_points = 8 * radius

lbp = local_binary_pattern(img, n_points, radius, method='uniform')

# Flatten features
features = lbp.reshape(-1, 1)

# KMeans clustering
kmeans = KMeans(n_clusters=2)
labels = kmeans.fit_predict(features)

segmented = labels.reshape(img.shape)

plt.imshow(segmented, cmap='gray')
plt.title("Texture Segmentation - LBP")
plt.show()
