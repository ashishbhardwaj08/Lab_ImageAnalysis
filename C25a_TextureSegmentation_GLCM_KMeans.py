

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


img = cv2.imread("Image.jpg", 0)

# Quantize image (reduce gray levels)
img_q = (img / 16).astype(np.uint8)

h, w = img.shape
window = 15

features = []

# Sliding window
for i in range(0, h-window, window):
    for j in range(0, w-window, window):
        patch = img_q[i:i+window, j:j+window]

        glcm = graycomatrix(patch,
                            distances=[1],
                            angles=[0],
                            levels=16,
                            symmetric=True,
                            normed=True)

        contrast = graycoprops(glcm, 'contrast')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]

        features.append([contrast, energy])

features = np.array(features)

# KMeans clustering
kmeans = KMeans(n_clusters=2)
labels = kmeans.fit_predict(features)

segmented = labels.reshape((h-window)//window,
                           (w-window)//window)

plt.imshow(segmented, cmap='gray')
plt.title("Texture Segmentation - GLCM")
plt.show()
