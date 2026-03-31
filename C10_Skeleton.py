import cv2
import numpy as np
from skimage.morphology import skeletonize

img = cv2.imread("Image5.png", 0)

# Threshold to get binary image
_, binary = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

# Ensure white objects on black background (invert if needed)
# distanceTransform works on WHITE regions
white_pixels = np.sum(binary == 255)
black_pixels = np.sum(binary == 0)
if white_pixels > black_pixels:
    binary = cv2.bitwise_not(binary)  # invert so objects are white

# Distance Transform on the binary image
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

# Normalize for display
dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
dist_norm = np.uint8(dist_norm)

# Find LOCAL MAXIMA (ridges) of distance map = true skeleton
# Dilate the distance map and find where original == dilated (local max)
kernel = np.ones((3, 3), np.uint8)
dist_dilated = cv2.dilate(dist, kernel, iterations=1)
local_max = (dist == dist_dilated)  # True at ridge/peak pixels

# Apply a minimum distance threshold to remove noise
min_dist_threshold = 0.1 * dist.max()  # at least 10% of max distance
skeleton = np.uint8(local_max & (dist > min_dist_threshold)) * 255

# Optional: thin further using skimage for 1-pixel-wide skeleton
binary_bool = binary > 0
skeleton_thin = skeletonize(binary_bool)
skeleton_thin = np.uint8(skeleton_thin) * 255

cv2.imshow("Original Binary", binary)
cv2.imshow("Distance Map", dist_norm)
cv2.imshow("Skeleton (Ridge)", skeleton)
cv2.imshow("Skeleton (Thinned)", skeleton_thin)
cv2.waitKey(0)
cv2.destroyAllWindows()