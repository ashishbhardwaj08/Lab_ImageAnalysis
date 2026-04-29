# Not Working Properly


import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Image.jpg")
template = cv2.imread("Template.jpg")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)


result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

# Find best match location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

h, w = template_gray.shape

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# Draw rectangle
matched = img.copy()
cv2.rectangle(matched, top_left, bottom_right, (255, 0, 0), 4)

cv2.imshow("Template", matched)


plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
plt.title("Template")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(result, cmap="gray")
plt.title("Matching Score Map")
plt.axis("off")


plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))
plt.title("Detected Template")
plt.axis("off")

plt.show()
