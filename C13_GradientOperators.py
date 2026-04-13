import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image as grayscale (0)
img = cv2.imread('Image5.png', 0)
# cv2.imshow("Original", img)



#  ROBERTS 
roberts_x = np.array([[1, 0],
                      [0, -1]], dtype=np.float32)

roberts_y = np.array([[0, 1],
                      [-1, 0]], dtype=np.float32)

rx = cv2.filter2D(img, -1, roberts_x)
ry = cv2.filter2D(img, -1, roberts_y)
roberts = cv2.magnitude(rx.astype(np.float32), ry.astype(np.float32))


#  PREWITT 
prewitt_x = np.array([[-1,0,1],
                      [-1,0,1],
                      [-1,0,1]], dtype=np.float32)

prewitt_y = np.array([[1,1,1],
                      [0,0,0],
                      [-1,-1,-1]], dtype=np.float32)

px = cv2.filter2D(img, -1, prewitt_x)
py = cv2.filter2D(img, -1, prewitt_y)
prewitt = cv2.magnitude(px.astype(np.float32), py.astype(np.float32))


#  SOBEL 
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)

# Convert to uint8
sobel = np.uint8(np.absolute(sobel))

# Display
# plt.imshow(sobel, cmap='gray')
# plt.title("Sobel Edge Detection")
# plt.show()


plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(roberts, cmap='gray')
plt.title("Roberts")

plt.subplot(1,3,2)
plt.imshow(prewitt, cmap='gray')
plt.title("Prewitt")

plt.subplot(1,3,3)
plt.imshow(sobel, cmap='gray')
plt.title("Sobel")

plt.show()



# cv2.imshow("Robert", roberts)
# cv2.imshow("Sobel", sobel)
# cv2.imshow("Prewitt", prewitt)



cv2.waitKey(0)
cv2.destroyAllWindows