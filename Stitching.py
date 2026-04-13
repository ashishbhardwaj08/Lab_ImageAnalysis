import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread("image.jpg")

# scale = 0.5
# img = cv2.resize(img, None, fx=scale, fy=scale)

# h, w = img.shape[:2]

# left_crop = img[:, :int(0.75 * w)]
# right_crop = img[:, int(0.25 * w):]

# cv2.imwrite("left_75.jpg", left_crop)
# cv2.imwrite("right_75.jpg", right_crop)

img1 = cv2.imread("left_75.jpg")
img2 = cv2.imread("right_75.jpg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(3000)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

good_matches = matches[:int(len(matches) * 0.2)]

similarity_score = len(good_matches) / len(matches)
print("Similarity Score:", similarity_score)

pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

result = cv2.warpPerspective(img2, H, (w1 + w2, h1))
result[0:h1, 0:w1] = img1

gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

h, w = gray.shape

last_col = w - 1
for col in range(w - 1, -1, -1):
    if np.any(gray[:, col] > 0):
        last_col = col
        break

cropped_result = result[:, :last_col + 1]

match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)

match_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
result_rgb = cv2.cvtColor(cropped_result, cv2.COLOR_BGR2RGB)



plt.figure(figsize=(20,10))

img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.subplot(2,2,1)
plt.imshow(img1_rgb)
plt.title("Feature Matches")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(img2_rgb)
plt.title("Feature Matches")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(match_rgb)
plt.title("Feature Matches")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(result_rgb)
plt.title("Final Stitched")
plt.axis("off")

plt.tight_layout()
plt.show()