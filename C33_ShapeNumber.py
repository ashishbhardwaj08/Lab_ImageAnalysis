# This code:

# Extracts contour
# Computes chain code
# Computes differential code
# Computes shape number
# Prints final result

import cv2
import numpy as np

# -------------------------------
# Direction Mapping (8-direction)
# -------------------------------
direction_dict = {
    (0, 1): 0,
    (-1, 1): 1,
    (-1, 0): 2,
    (-1, -1): 3,
    (0, -1): 4,
    (1, -1): 5,
    (1, 0): 6,
    (1, 1): 7
}

# -------------------------------
# Generate Chain Code
# -------------------------------
def generate_chain_code(contour):
    chain_code = []
    for i in range(len(contour) - 1):
        x1, y1 = contour[i][0]
        x2, y2 = contour[i+1][0]
        dx = x2 - x1
        dy = y2 - y1
        if (dx, dy) in direction_dict:
            chain_code.append(direction_dict[(dx, dy)])
    return chain_code

# -------------------------------
# Differential Chain Code
# -------------------------------
def differential_chain_code(chain):
    diff = []
    for i in range(len(chain)):
        diff.append((chain[(i+1)%len(chain)] - chain[i]) % 8)
    return diff

# -------------------------------
# Compute Shape Number
# -------------------------------
def shape_number(diff_chain):
    n = len(diff_chain)
    min_seq = diff_chain
    
    for i in range(1, n):
        rotated = diff_chain[i:] + diff_chain[:i]
        if rotated < min_seq:
            min_seq = rotated
    
    return min_seq

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    
    img = cv2.imread("Image5.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    
    contour = contours[0]
    
    chain = generate_chain_code(contour)
    diff_chain = differential_chain_code(chain)
    sn = shape_number(diff_chain)
    
    print("Chain Code (first 50):")
    print(chain[:50])
    
    print("\nDifferential Chain Code (first 50):")
    print(diff_chain[:50])
    
    print("\nShape Number (first 50):")
    print(sn[:50])
