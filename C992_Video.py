
import cv2
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (640, 480))

    cv2.imshow("Webcam", frame)

    # Esc key to close program
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()