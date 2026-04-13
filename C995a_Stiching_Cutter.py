
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Stitch1.jpeg")

scale = 0.5
img = cv2.resize(img, None, fx=scale, fy=scale)

h, w = img.shape[:2]

left_crop = img[:, :int(0.75 * w)]

right_crop = img[:, int(0.25 * w):]

#NOTE: OpenCV uses BGR format, but Matplotlib uses RGB, so we need to convert before displaying
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
left_rgb = cv2.cvtColor(left_crop, cv2.COLOR_BGR2RGB)
right_rgb = cv2.cvtColor(right_crop, cv2.COLOR_BGR2RGB)


plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(left_rgb)
plt.title("Left 75%")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(right_rgb)
plt.title("Right 75%")
plt.axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite("left_75.jpg", left_crop)
cv2.imwrite("right_75.jpg", right_crop)