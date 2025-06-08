import cv2

for device_id in range(0, 10):
    cap = cv2.VideoCapture(device_id)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f'Successfully grabbed frame from /dev/video{device_id}')
            cv2.imshow('test', frame)
            cv2.waitKey(3000)
            cap.release()
            cv2.destroyAllWindows()
            break
        cap.release()
print("Done checking devices")