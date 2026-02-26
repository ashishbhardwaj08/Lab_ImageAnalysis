import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
import matplotlib.pyplot as plt

# -----------------------------
# 1. Read and Convert to Binary
# -----------------------------
image = cv2.imread("input.png", 0)   # grayscale
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Convert to boolean (required for skimage)
binary_bool = binary // 255

# -----------------------------
# 2. Skeletonization
# -----------------------------
skeleton = skeletonize(binary_bool)

# Convert back to uint8
skeleton_uint8 = (skeleton * 255).astype(np.uint8)

# -----------------------------
# 3. Pruning (Remove Small Branches)
# -----------------------------
# Remove small connected components from skeleton
pruned = remove_small_objects(skeleton, min_size=30)

pruned_uint8 = (pruned * 255).astype(np.uint8)

# -----------------------------
# 4. Show Results
# -----------------------------
plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.title("Binary")
plt.imshow(binary, cmap='gray')

plt.subplot(1,3,2)
plt.title("Skeleton")
plt.imshow(skeleton_uint8, cmap='gray')

plt.subplot(1,3,3)
plt.title("Pruned Skeleton")
plt.imshow(pruned_uint8, cmap='gray')

plt.tight_layout()
plt.show()
