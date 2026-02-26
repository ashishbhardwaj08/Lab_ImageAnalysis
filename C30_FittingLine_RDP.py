#Error: Currently only works on one shape

import cv2
import numpy as np

# -------------------------------
# Ramer-Douglas-Peucker Algorithm
# -------------------------------
def rdp(points, epsilon):
    if len(points) < 3:
        return points
    
    # Line from first to last point
    start = points[0]
    end = points[-1]
    
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    
    max_dist = 0
    index = 0
    
    for i in range(1, len(points)-1):
        point = points[i]
        
        # Distance formula
        dist = np.abs(np.cross(line_vec, start-point)) / line_len
        
        if dist > max_dist:
            max_dist = dist
            index = i
    
    # Split if needed
    if max_dist > epsilon:
        left = rdp(points[:index+1], epsilon)
        right = rdp(points[index:], epsilon)
        
        return np.vstack((left[:-1], right))
    else:
        return np.array([start, end])

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    
    img = cv2.imread("Image5.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    
    contour = contours[0]
    
    # Convert contour to Nx2 format
    points = contour.reshape(-1,2)
    
    # Apply RDP
    epsilon = 5.0   # tolerance
    simplified = rdp(points, epsilon)
    
    # Draw original contour
    img_copy = img.copy()
    cv2.drawContours(img_copy, contours, -1, (0,255,0), 1)
    
    # Draw simplified polygon
    for i in range(len(simplified)-1):
        pt1 = tuple(simplified[i])
        pt2 = tuple(simplified[i+1])
        cv2.line(img_copy, pt1, pt2, (0,0,255), 2)
    
    print("Original contour points:", len(points))
    print("Simplified points:", len(simplified))
    
    cv2.imshow("Polygonal Approximation", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
