import cv2

img1 = cv2.imread('left_75.jpg')
img2 = cv2.imread('right_75.jpg')

stitcher = cv2.Stitcher_create()

status, pano = stitcher.stitch([img1, img2])

if status == cv2.Stitcher_OK:
    cv2.imshow('Panorama', pano)
    cv2.waitKey(0)
