import cv2
import numpy as np

img = cv2.imread("Image5.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (9, 9), 2)

circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=20,
    param1=50,
    param2=30,
    minRadius=0,
    maxRadius=200,
)


if circles is not None:
    circles = np.uint16(np.around(circles))

    for circle in circles[0, :]:
        x, y, r = circle

        # Draw outer circle
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)

        # Draw center point
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)


cv2.imshow("Detected Circles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
