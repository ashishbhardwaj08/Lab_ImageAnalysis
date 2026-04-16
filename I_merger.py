import cv2
import numpy as np

# Read masks (grayscale)
mask1 = cv2.imread('km_American_Crow_0014_25287.png', cv2.IMREAD_GRAYSCALE)
mask2 = cv2.imread('ot_American_Crow_0014_25287.png', cv2.IMREAD_GRAYSCALE)

merged = cv2.addWeighted(mask1, 0.5, mask2, 0.5, 0)

_,binary = cv2.threshold(merged, 135, 255, cv2.THRESH_BINARY)

closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

# Remove small white patches using connected components
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closing, connectivity=8)

min_area = 500  # adjust this based on how small patches are
cleaned = np.zeros_like(closing)

for i in range(1, num_labels):  # skip background (0)
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= min_area:
        cleaned[labels == i] = 255


cv2.imshow('binary.png', binary)

cv2.imshow('merged_mask.png', closing)

cv2.imshow('cleaned_mask.png', cleaned)

cv2.waitKey(0)
cv2.destroyAllWindows()