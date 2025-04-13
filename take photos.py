import cv2
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
frame_count=0
try:
    while True:
        frame_count = frame_count+1
        ret,frame=cap.read()
        print(f'{frame.shape}')
        if not ret:
            break
        """cv2.imshow('now',frame)"""
        cv2.imwrite(r"C:\Users\qixin\Desktop\result\frame_{}.jpg".format(frame_count), frame)
finally:
    # 释放资源
    cap.release()
