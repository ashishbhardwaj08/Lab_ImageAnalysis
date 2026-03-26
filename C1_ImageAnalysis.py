import cv2
import numpy as np
from matplotlib import pyplot as plt

original = cv2.imread("Image5.png")


print("img shape:", original.shape)

#h, w, c = original.shape
#print("Width:", w)
#print("Height:", h)
#print("Channels:", c)


img = cv2.resize(original, (640,427))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127,255, cv2.THRESH_BINARY)

rotated = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
avg_blur = cv2.blur(img, (15,15))
gauss_blur = cv2.GaussianBlur(img, (5,5), 0)
median_blur = cv2.medianBlur(img, 5)





edges = cv2.Canny(img, 100, 200)


cropped = img[250:750, 250:500]
# cv2.imwrite("Template.jpg",cropped)


flipped = cv2.flip(img, 0)# 1- Horizontal, 0 - Vertical

contrast = cv2.convertScaleAbs(img, alpha=2, beta=0)


draw_img = img.copy()
cv2.line(draw_img, (0,0), (640,427), (0,255,0), 3)
cv2.line(draw_img, (0,427), (640,0), (0,255,0), 3)
cv2.rectangle(draw_img, (220,113),(420,313), (255,255,255), 4)
cv2.circle(draw_img, (320, 213), 145, (255,255,0), 4)
cv2.putText(draw_img, "OpenCV", (50,50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)



kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
sharpen = cv2.filter2D(img, -1, kernel)

invert = cv2.bitwise_not(img)

gamma = 2.0
gamma_corrected = np.power(img/255.0, gamma)
gamma_corrected = np.uint8(gamma_corrected*255)


original2 = cv2.imread("Image2.jpg")
img2 = cv2.resize(original2,(640,427) )


and_img = cv2.bitwise_and(img, img2)
or_img  = cv2.bitwise_or(img, img2)
xor_img = cv2.bitwise_xor(img, img2)

added = cv2.add(img, img2)

subtracted = cv2.subtract(img, img2)

overlay = img.copy()
small = cv2.resize(img2, (200,200))
overlay[0:200, 0:200] = small

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Mask specific color (detect blue color)
lower_blue = np.array([100,150,0])
upper_blue = np.array([140,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
color_detected = cv2.bitwise_and(img, img, mask=mask)

negative = 255 - img

'''plt.hist(img.ravel(), 256, [0,256])
plt.title("Histogram")
plt.show()'''

normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

zoom_in = cv2.resize(img, None, fx=2, fy=2)
zoom_out = cv2.resize(img, None, fx=0.5, fy=0.5)


b, g, r = cv2.split(img)

"""cv2.imshow("Blue Channel", b)
cv2.imshow("Green Channel", g)
cv2.imshow("Red Channel", r)
"""

zero = np.zeros_like(b)

merged = cv2.merge([b, g, zero])


#cv2.imshow("Merged",merged)

cv2.imshow("File",gray)
cv2.imshow("File2",binary)




cv2.waitKey(0)
# cv2.imwrite("Binary.jpg", gray)
cv2.destroyAllWindows()