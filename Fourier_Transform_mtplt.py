import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread('Image.jpg', 0)

# Apply Fourier Transform
f = np.fft.fft2(img)

# Shift zero frequency to center
fshift = np.fft.fftshift(f)

# Magnitude spectrum
magnitude = 20 * np.log(np.abs(fshift) + 1)

# Inverse Fourier Transform
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Display results
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.axis('off')

plt.subplot(132), plt.imshow(magnitude, cmap='gray')
plt.title('Magnitude Spectrum'), plt.axis('off')

plt.subplot(133), plt.imshow(img_back, cmap='gray')
plt.title('Reconstructed Image'), plt.axis('off')

plt.show()
