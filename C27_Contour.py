import cv2
import numpy as np

img = cv2.imread("Image5.png", 0)


_, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(binary,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

# Draw contours
output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output, contours, -1, (0,0,255), 2)

print("Number of contours found:", len(contours))

cv2.imshow("Contour Following", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
