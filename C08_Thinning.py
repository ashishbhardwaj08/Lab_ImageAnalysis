import cv2
import numpy as np

img = cv2.imread("Image5.png", 0)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

binary = binary // 255  # Convert to 0 and 1

skeleton = np.zeros(binary.shape, np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

temp = np.zeros(binary.shape, np.uint8)

while True:
    eroded = cv2.erode(binary, kernel)
    opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
    temp = eroded - opened
    skeleton = cv2.bitwise_or(skeleton, temp)
    binary = eroded.copy()

    if cv2.countNonZero(binary) == 0:
        break

skeleton = skeleton * 255

cv2.imshow("Skeleton", skeleton)
cv2.waitKey(0)
