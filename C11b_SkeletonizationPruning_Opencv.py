import cv2
import numpy as np

# -----------------------------
# 1. Read Image and Convert to Binary
# -----------------------------
img = cv2.imread("Image5.png", 0)

_, binary = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

# Invert if background is white (optional)
# binary = cv2.bitwise_not(binary)

# -----------------------------
# 2. Skeletonization (Morphological Method)
# -----------------------------
skeleton = np.zeros(binary.shape, np.uint8)

element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
temp = np.zeros(binary.shape, np.uint8)
eroded = np.zeros(binary.shape, np.uint8)

while True:
    eroded = cv2.erode(binary, element)
    opened = cv2.dilate(eroded, element)
    temp = cv2.subtract(binary, opened)
    skeleton = cv2.bitwise_or(skeleton, temp)
    binary = eroded.copy()

    if cv2.countNonZero(binary) == 0:
        break

# -----------------------------
# 3. Pruning (Remove Small Branches)
# -----------------------------
# Find connected components
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)

min_size = 30   # change this to control pruning

pruned = np.zeros_like(skeleton)

for i in range(1, num_labels):  # skip background
    if stats[i, cv2.CC_STAT_AREA] >= min_size:
        pruned[labels == i] = 255

# -----------------------------
# 4. Show Results
# -----------------------------
cv2.imshow("Binary", img)
cv2.imshow("Skeleton", skeleton)
cv2.imshow("Pruned Skeleton", pruned)

cv2.waitKey(0)
cv2.destroyAllWindows()
