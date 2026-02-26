# Creates a test shape
# Extracts contour
# Converts to complex sequence
# Computes AR coefficients
# Reconstructs shape
# Plots comparison

import numpy as np
import cv2
import matplotlib.pyplot as plt

# ------------------------------
# 1. Generate Binary Shape
# ------------------------------
img = np.zeros((300,300), dtype=np.uint8)
cv2.circle(img, (150,150), 80, 255, -1)

# ------------------------------
# 2. Extract Contour
# ------------------------------
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour = contours[0]

# Convert contour to complex sequence
z = contour[:,0,0] + 1j * contour[:,0,1]
z = z.astype(np.complex128)

# ------------------------------
# 3. AR Model Estimation
# ------------------------------
def estimate_ar(z, p):
    N = len(z)
    X = []
    Y = []
    
    for i in range(p, N):
        X.append(z[i-p:i])
        Y.append(z[i])
    
    X = np.array(X)
    Y = np.array(Y)
    
    # Solve least squares
    a = np.linalg.lstsq(X, Y, rcond=None)[0]
    return a

# Model order
p = 10
a = estimate_ar(z, p)

print("AR Coefficients:")
print(a)

# ------------------------------
# 4. Shape Reconstruction
# ------------------------------
z_recon = np.copy(z[:p])

for i in range(p, len(z)):
    val = np.dot(a, z_recon[i-p:i])
    z_recon = np.append(z_recon, val)

# ------------------------------
# 5. Plot Results
# ------------------------------
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.plot(z.real, z.imag)
plt.title("Original Shape")
plt.gca().invert_yaxis()

plt.subplot(1,2,2)
plt.plot(z_recon.real, z_recon.imag)
plt.title("AR Reconstructed Shape")
plt.gca().invert_yaxis()

plt.show()
# Step 1:
# Binary image created.
# Step 2:
# Contour extracted as ordered boundary.
# Step 3:
# Least squares used to estimate AR coefficients.
# Step 4:
# Reconstruction using learned model.
# Step 5:
# Comparison plotted.