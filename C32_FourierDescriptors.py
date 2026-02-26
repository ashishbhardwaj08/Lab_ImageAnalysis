# WHAT THIS CODE DOES

# Converts boundary to frequency domain
# Keeps only 20 descriptors
# Removes noise
# Reconstructs smooth boundary

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    
    # Read image
    img = cv2.imread("Image5.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    
    contour = contours[0]
    points = contour.reshape(-1,2)
    
    # Convert to complex numbers
    complex_points = points[:,0] + 1j * points[:,1]
    
    # Fourier Transform
    descriptors = np.fft.fft(complex_points)
    
    # Normalize (scale invariance)
    descriptors = descriptors / np.abs(descriptors[1])
    
    # Remove high frequency components
    keep = 20   # number of coefficients to keep
    descriptors[keep:-keep] = 0
    
    # Inverse FFT to reconstruct
    reconstructed = np.fft.ifft(descriptors)
    
    x_rec = reconstructed.real
    y_rec = reconstructed.imag
    
    # Plot
    plt.figure()
    plt.imshow(binary, cmap='gray')
    plt.plot(points[:,0], points[:,1], 'r.', label="Original")
    plt.plot(x_rec, y_rec, 'b-', linewidth=2, label="Reconstructed")
    plt.legend()
    plt.title("Fourier Descriptor Reconstruction")
    plt.show()
