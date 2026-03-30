import cv2
import numpy as np

# Load image
img = cv2.imread("Dazai1.jpg")
# img = cv2.resize(img, (512, 512))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def nothing(x):
    pass

# Create control window
cv2.namedWindow("Controls")

# Sliders
cv2.createTrackbar("Spacing", "Controls", 10, 30, nothing)
cv2.createTrackbar("Blur", "Controls", 3, 21, nothing)
cv2.createTrackbar("Thresh1", "Controls", 80, 255, nothing)
cv2.createTrackbar("Thresh2", "Controls", 150, 255, nothing)
cv2.createTrackbar("Thresh3", "Controls", 200, 255, nothing)

while True:
    # Get slider values
    spacing = cv2.getTrackbarPos("Spacing", "Controls")
    blur_val = cv2.getTrackbarPos("Blur", "Controls")
    t1 = cv2.getTrackbarPos("Thresh1", "Controls")
    t2 = cv2.getTrackbarPos("Thresh2", "Controls")
    t3 = cv2.getTrackbarPos("Thresh3", "Controls")

    # Fix blur (must be odd)
    if blur_val % 2 == 0:
        blur_val += 1
    if blur_val < 1:
        blur_val = 1

    # Smooth image
    smooth = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)

    # Normalize
    norm = smooth / 255.0
    h, w = norm.shape

    # Canvas
    hatch = np.ones((h, w), dtype=np.float32)

    # Create patterns
    def pattern_diag1():
        p = np.zeros((h, w))
        for i in range(0, h, spacing):
            for j in range(w):
                if (i + j) < h:
                    p[i + j, j] = 1
        return p

    def pattern_diag2():
        p = np.zeros((h, w))
        for i in range(0, h, spacing):
            for j in range(w):
                if (i - j) >= 0:
                    p[i - j, j] = 1
        return p

    def pattern_h():
        p = np.zeros((h, w))
        for i in range(0, h, spacing):
            p[i:i+1, :] = 1
        return p

    def pattern_v():
        p = np.zeros((h, w))
        for i in range(0, w, spacing):
            p[:, i:i+1] = 1
        return p

    p1 = pattern_diag1()
    p2 = pattern_diag2()
    p3 = pattern_h()
    p4 = pattern_v()

    # Apply layers based on thresholds
    for i in range(h):
        for j in range(w):
            val = smooth[i, j]

            if val < t3:
                hatch[i, j] *= (1 - p1[i, j])
            if val < t2:
                hatch[i, j] *= (1 - p2[i, j])
            if val < t1:
                hatch[i, j] *= (1 - p3[i, j])
            if val < t1 // 2:
                hatch[i, j] *= (1 - p4[i, j])

    hatch = (hatch * 255).astype(np.uint8)

    # Edge overlay (optional but looks great)
# edges = cv2.Canny(smooth, 50, 150)
    # final = cv2.bitwise_and(hatch, hatch, mask=edges)

    # Show results
    # cv2.imshow("Original", img)
    cv2.imshow("Hatching", hatch)
    # cv2.imshow("Final (Hatch + Edges)", final)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()