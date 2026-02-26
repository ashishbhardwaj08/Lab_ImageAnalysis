import cv2
import numpy as np

img = cv2.imread("Image5.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150)

lines = cv2.HoughLines(edges,
                       1,            # rho resolution
                       np.pi/180,    # theta resolution
                       150)          # threshold


#Probabalistic Hough Transform 
#More Efficient

# lines = cv2.HoughLinesP(edges,
#                         1,
#                         np.pi/180,
#                         100,
#                         minLineLength=100,
#                         maxLineGap=10)




for rho, theta in lines[:,0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
    

cv2.imshow("Hough Lines", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
