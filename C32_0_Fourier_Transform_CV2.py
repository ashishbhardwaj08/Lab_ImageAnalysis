import cv2
import numpy as np

# Read image in grayscale
img = cv2.imread('Image.jpg', 0)

# Apply Fourier Transform
f = np.fft.fft2(img)

# Shift zero frequency to center
fshift = np.fft.fftshift(f)

# Magnitude spectrum
magnitude = 20 * np.log(np.abs(fshift) + 1)

# Normalize for display
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
magnitude = magnitude.astype(np.uint8)

# Inverse Fourier Transform
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
img_back = np.uint8(img_back)

# Display results
cv2.imshow("Original Image", img)
cv2.imshow("Magnitude Spectrum", magnitude)
cv2.imshow("Reconstructed Image", img_back)

cv2.waitKey(0)
cv2.destroyAllWindows()