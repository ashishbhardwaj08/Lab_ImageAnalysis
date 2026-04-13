import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread("Image.jpg", 0)

# KIRSCH OPERATOR

kirsch_masks = [
    np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]]),      # N
    np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]]),      # NE
    np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]]),      # E
    np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]]),      # SE
    np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]]),      # S
    np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]]),      # SW
    np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]]),      # W
    np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]])       # NW
]

kirsch_responses = []

for mask in kirsch_masks:
    response = cv2.filter2D(img, cv2.CV_64F, mask)
    kirsch_responses.append(np.abs(response))

kirsch_edge = np.max(np.stack(kirsch_responses), axis=0)
kirsch_edge = np.uint8(np.clip(kirsch_edge, 0, 255))



# ROBINSON OPERATOR


robinson_masks = [
    np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),   # E
    np.array([[0,1,2],[-1,0,1],[-2,-1,0]]),   # NE
    np.array([[1,2,1],[0,0,0],[-1,-2,-1]]),   # N
    np.array([[2,1,0],[1,0,-1],[0,-1,-2]]),   # NW
    np.array([[1,0,-1],[2,0,-2],[1,0,-1]]),   # W
    np.array([[0,-1,-2],[1,0,-1],[2,1,0]]),   # SW
    np.array([[-1,-2,-1],[0,0,0],[1,2,1]]),   # S
    np.array([[-2,-1,0],[-1,0,1],[0,1,2]])    # SE
]

robinson_responses = []

for mask in robinson_masks:
    response = cv2.filter2D(img, cv2.CV_64F, mask)
    robinson_responses.append(np.abs(response))

robinson_edge = np.max(np.stack(robinson_responses), axis=0)
robinson_edge = np.uint8(np.clip(robinson_edge, 0, 255))






plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(kirsch_edge, cmap='gray')
plt.title("Kirsch Edge Detection")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(robinson_edge, cmap='gray')
plt.title("Robinson Edge Detection")
plt.axis("off")

plt.tight_layout()
plt.show()
