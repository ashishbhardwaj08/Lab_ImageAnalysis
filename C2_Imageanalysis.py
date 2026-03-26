import cv2
import numpy as np

img = cv2.imread("Image4.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Edge detection (Canny)
edges = cv2.Canny(gray, 100, 200)

# Sobel filter
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

# Prewitt filter
kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
prewittx = cv2.filter2D(gray, -1, kernelx)
prewitty = cv2.filter2D(gray, -1, kernely)

# Laplacian filter
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# Morphological operations (erode, dilate)
kernel = np.ones((5,5), np.uint8)
erode = cv2.erode(gray, kernel, iterations=1)
dilate = cv2.dilate(gray, kernel, iterations=1)

# Opening
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

# Closing
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

# Find contours
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)

# Object area calculation
'''for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 50:
        print("Area:", area)
'''

# cnt in contours:
# area = cv2.contourArea(cnt)
# perimeter = cv2.arcLength(cnt, True)
# print("Area:", area, "Perimeter:", perimeter)

# Bounding box detection
bbox_img = img.copy()
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(bbox_img, (x,y), (x+w,y+h), (255,0,0), 2)

# Circle detection (Hough Circle)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                           param1=50, param2=30, minRadius=10, maxRadius=100)
circle_img = img.copy()
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0]:
        cv2.circle(circle_img, (i[0], i[1]), i[2], (0,255,0), 2)

# Line detection (Hough Line)
lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
line_img = img.copy()
if lines is not None:
    for i in lines[:10]:
        rho, theta = i[0]
        a = np.cos(theta); b = np.sin(theta)
        x0 = a*rho; y0 = b*rho
        x1 = int(x0 + 1000*(-b)); y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b)); y2 = int(y0 - 1000*(a))
        cv2.line(line_img, (x1,y1), (x2,y2), (0,0,255), 2)

# Background subtraction
bg = cv2.createBackgroundSubtractorMOG2()
fgmask = bg.apply(img)

# Image segmentation (simple threshold)
seg = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                             cv2.THRESH_BINARY,11,2)

# Adaptive thresholding
adaptive = cv2.adaptiveThreshold(gray,255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY,11,2)

# Noise removal
noise_removed = cv2.medianBlur(gray, 5)

# Image pyramids
lower = cv2.pyrDown(img)
higher = cv2.pyrUp(lower)

# Image blending
img2 = cv2.imread("Image2.jpg")
img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
blend = cv2.addWeighted(img, 0.7, img2, 0.3, 0)

# Template matching
template = cv2.imread("template.jpg", 0)
res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
_, _, _, max_loc = cv2.minMaxLoc(res)
h, w = template.shape
tm_img = img.copy()
cv2.rectangle(tm_img, max_loc, (max_loc[0]+w, max_loc[1]+h), (0,255,0), 2)

# Perspective transform
pts1 = np.float32([[50,50],[200,50],[50,200],[200,200]])
pts2 = np.float32([[10,100],[200,50],[100,250],[300,200]])
M = cv2.getPerspectiveTransform(pts1, pts2)
perspective = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

# Affine transform
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M_aff = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(img, M_aff, (img.shape[1], img.shape[0]))

# Image warping
warp = cv2.warpAffine(img, M_aff, (img.shape[1], img.shape[0]))

# Region of Interest (ROI) / Cropping
roi = img[100:300, 200:400]

# lay key results
cv2.imshow("imgout", opening)


cv2.waitKey(0)
cv2.destroyAllWindows()
