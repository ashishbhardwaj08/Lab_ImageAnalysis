import cv2

img = cv2.imread("Image.jpg", 0)  # 0 = grayscale

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

cv2.imwrite("Binary.jpg", binary)
cv2.waitKey(0)
cv2.destroyAllWindows

