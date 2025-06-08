import cv2
import numpy as np

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Cannot open camera...")
    exit()
while True():
    # Frame by frame
    ret, frame = capture.read()

    if not ret:
        print("Cannot recieve frame (stream end?). Exiting...")
        break

    # Frame operations
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break

# When all done, release capture
capture.release()
cv2.destroyAllWindows()
