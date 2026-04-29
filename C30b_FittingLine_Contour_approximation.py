# This also uses RDP internally


import cv2
import numpy as np

img = cv2.imread("Image5.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Loop through each contour
for contour in contours:

    # -----------------------------
    # 4. Polygon Approximation
    # -----------------------------
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Draw original contour (White)
    cv2.drawContours(img, [contour], -1, (255, 255, 255), 1)

    # Draw approximated contour (Green)
    cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)

    # Optional: Identify shape
    vertices = len(approx)

    if vertices == 3:
        shape = "Triangle"
    elif vertices == 4:
        shape = "Quadrilateral"
    elif vertices > 4:
        shape = "Circle/Polygon"
    else:
        shape = "Unknown"

    # Put text at contour center
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(
            img, shape, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )


cv2.imshow("Polygon Approximation", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
