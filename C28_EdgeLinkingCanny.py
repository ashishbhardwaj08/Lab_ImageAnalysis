import cv2
import numpy as np

img = cv2.imread("image.jpg", 0)

edges = cv2.Canny(img, 50, 150)

cv2.imshow("Edge Linking using Canny", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
