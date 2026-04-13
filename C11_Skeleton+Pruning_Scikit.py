import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
import matplotlib.pyplot as plt


image = cv2.imread("Image.png", 0)   # grayscale
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Convert to boolean (required for skimage)
binary_bool = binary // 255

# Skeletonization
skeleton = skeletonize(binary_bool)

# Convert back to uint8
skeleton_uint8 = (skeleton * 255).astype(np.uint8)

# Pruning (Remove Small Branches)
# Remove small connected components from skeleton
pruned = remove_small_objects(skeleton, max_size=2)

pruned_uint8 = (pruned * 255).astype(np.uint8)

# Show Results

cv2.imshow("Skeleton", skeleton_uint8)
cv2.imshow("Skeleton Pruned", pruned_uint8)



# plt.figure(figsize=(10,5))

# plt.subplot(1,3,1)
# plt.title("Binary")
# plt.imshow(binary, cmap='gray')

# plt.subplot(1,3,2)
# plt.title("Skeleton")
# plt.imshow(skeleton_uint8, cmap='gray')

# plt.subplot(1,3,3)
# plt.title("Pruned Skeleton")
# plt.imshow(pruned_uint8, cmap='gray')

# plt.tight_layout()
# plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()