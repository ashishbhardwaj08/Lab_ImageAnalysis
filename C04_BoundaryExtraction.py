import cv2
import numpy as np

img = cv2.imread("Dazai1.jpg", 0)
_, binary = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

kernel = np.ones((3,3), np.uint8)

eroded = cv2.erode(binary, kernel, iterations=1)

boundary = cv2.subtract(binary, eroded)

cv2.imshow("Original", binary)
cv2.imshow("Eroded", eroded)
cv2.imshow("Boundary", boundary)
cv2.waitKey(0)


































# import cv2
# import numpy as np

# img = cv2.imread("Dazai1.jpg", 0)

# def nothing(x):
#     pass
# cv2.namedWindow("Controls")
# cv2.createTrackbar("Blur", "Controls", 1, 255, nothing)
# cv2.createTrackbar("Iterations", "Controls", 1, 10, nothing)


# while True:
#     # Get slider values
#     blur_val = cv2.getTrackbarPos("Blur", "Controls")
#     iter_val = cv2.getTrackbarPos("Iterations", "Controls")

#     # Blur must be odd
#     if blur_val % 2 == 0:
#         blur_val += 1
        
#     _, binary = cv2.threshold(img, blur_val, 255, cv2.THRESH_BINARY)

#     kernel = np.ones((3,3), np.uint8)

#     eroded = cv2.erode(binary, kernel, iterations=iter_val)

#     boundary = cv2.subtract(binary, eroded)

#     cv2.imshow("Original", binary)
#     cv2.imshow("Eroded", eroded)
#     cv2.imshow("Boundary", 255 - boundary)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
    



