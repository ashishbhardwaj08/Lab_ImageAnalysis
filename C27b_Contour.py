


# NOT WORKING CORRECTLY.  JUST FOR MATHEMATICAL IDEA
# Detects only one shape starting from top right





import cv2
import numpy as np

# -----------------------------
# Neighbor definition (8-connected)
# -----------------------------
def get_neighbors(x, y):
    return [
        (x-1,y-1),(x-1,y),(x-1,y+1),
        (x,y+1),
        (x+1,y+1),(x+1,y),(x+1,y-1),
        (x,y-1)
    ]

# -----------------------------
# Contour following function
# -----------------------------
def contour_follow(binary, start):
    contour = []
    current = start
    visited = set()

    h, w = binary.shape

    while True:
        contour.append(current)
        visited.add(current)

        found_next = False

        for n in get_neighbors(*current):
            x, y = n

            # check image bounds
            if x < 0 or y < 0 or x >= h or y >= w:
                continue

            # check white pixel and not already visited
            if binary[x, y] == 255 and n not in visited:
                current = n
                found_next = True
                break

        # stop if closed loop or no next pixel
        if not found_next or current == start:
            break

    return contour

# -----------------------------
# Main Program
# -----------------------------
img = cv2.imread("Image5.png")  # <-- give your image path here
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to binary
_, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# -----------------------------
# Find starting boundary pixel
# -----------------------------
start = None
h, w = binary.shape

for i in range(h):
    for j in range(w):
        if binary[i, j] == 255:
            start = (i, j)
            break
    if start is not None:
        break

# -----------------------------
# Get contour
# -----------------------------
contour = contour_follow(binary, start)

# -----------------------------
# Draw contour
# -----------------------------
output = img.copy()

for (x, y) in contour:
    cv2.circle(output, (y, x), 1, (0, 0, 255), -1)

# -----------------------------
# Show results
# -----------------------------
cv2.imshow("Original", img)
cv2.imshow("Binary", binary)
cv2.imshow("Contour", output)

cv2.waitKey(0)
cv2.destroyAllWindows()