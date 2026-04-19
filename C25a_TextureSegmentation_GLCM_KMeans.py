import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

img = cv2.imread("Image5.png", 0)

# Quantization (16 gray levels)
img_q = (img / 16).astype(np.uint8)

h, w = img.shape
window = 15

features = []

rows = h // window
cols = w // window

# Sliding window
for i in range(0, rows * window, window):
    for j in range(0, cols * window, window):

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
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
labels = kmeans.fit_predict(features)

segmented = labels.reshape(rows, cols)

plt.imshow(segmented, cmap='gray')
plt.title("Texture Segmentation - GLCM")
plt.colorbar()
plt.show()