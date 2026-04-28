import cv2
import numpy as np

img = cv2.imread("Image.png", 0)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Convert to 0/1
binary = binary // 255

B1 = np.array([[0,1,0],
               [0,1,0],
               [0,0,0]], dtype=np.uint8)

B2 = np.array([[1,0,1],
               [1,0,1],
               [1,1,1]], dtype=np.uint8)

def hit_or_miss(img, B1, B2):
    erode1 = cv2.erode(img, B1)
    erode2 = cv2.erode(1-img, B2)
    return erode1 & erode2

prev = np.zeros_like(binary)

while True:
    hm = hit_or_miss(binary, B1, B2)
    binary = binary | hm
    
    if np.array_equal(binary, prev):
        break
    
    prev = binary.copy()

thickened = binary * 255

cv2.imshow("Thickened (Hit-or-Miss)", thickened)
cv2.waitKey(0)
