import cv2
import numpy as np

# Load image
img = cv2.imread("Dazai2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Dummy function (required for trackbars)
def nothing(x):
    pass

# Create window
cv2.namedWindow("Controls")

# Create sliders
cv2.createTrackbar("Blur", "Controls", 21, 101, nothing)
cv2.createTrackbar("Canny Min", "Controls", 50, 255, nothing)
cv2.createTrackbar("Canny Max", "Controls", 150, 255, nothing)

while True:
    # Get slider values
    blur_val = cv2.getTrackbarPos("Blur", "Controls")
    canny_min = cv2.getTrackbarPos("Canny Min", "Controls")
    canny_max = cv2.getTrackbarPos("Canny Max", "Controls")

    # Blur must be odd
    if blur_val % 2 == 0:
        blur_val += 1

    # -------- Pencil Sketch --------
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (blur_val, blur_val), 0)
    inv_blur = 255 - blur
    sketch = cv2.divide(gray, inv_blur, scale=256.0)

    # -------- Pen Drawing --------
    edges = cv2.Canny(gray, canny_min, canny_max)
    pen = 255 - edges
    # Show outputs
    # cv2.imshow("Original", img)
    cv2.imshow("Sketch", sketch)
    cv2.imshow("Pen Drawing", pen)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()