import cv2
import numpy as np

img = cv2.imread("Image5.png", 0)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel1 = np.array([[1,1,1],
                    [0,1,0],
                    [0,0,0]], dtype=np.uint8)

kernel2 = np.array([[0,0,0],
                    [1,0,1],
                    [1,1,1]], dtype=np.uint8)

erode1 = cv2.erode(binary, kernel1)

inv = cv2.bitwise_not(binary)
erode2 = cv2.erode(inv, kernel2)

hitmiss = cv2.bitwise_and(erode1, erode2)

cv2.imwrite("Hit-or-Miss.jpg", hitmiss)
cv2.waitKey(0)
