import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    
    # Read image
    img = cv2.imread("Image5.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    
    contour = contours[0]
    
    # Convert contour to Nx2
    points = contour.reshape(-1,2)
    
    # Sample points to reduce density
    sampled = points[::10]
    
    x = sampled[:,0]
    y = sampled[:,1]
    
    # B-spline fitting
    tck, u = splprep([x, y], s=5, k=3)
    
    u_new = np.linspace(0, 1, 1000)
    x_new, y_new = splev(u_new, tck)
    
    # Plot
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.plot(x, y, 'ro', label='Control Points')
    plt.plot(x_new, y_new, 'b-', linewidth=2, label='B-Spline Curve')
    plt.legend()
    plt.title("B-Spline Boundary Representation")
    plt.show()
