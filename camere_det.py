"""import cv2

cap = cv2.VideoCapture(0)  # 修改为你的摄像头索引
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头！")
        break
    cv2.imshow('Test', frame)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()"""
import cv2

def show_camera_feed(camera_index=1):
    """
    显示指定索引的摄像头画面。
    :param camera_index: 摄像头索引，默认为 0（通常表示默认摄像头）。
    """
    # 打开摄像头
    cap = cv2.VideoCapture(camera_index,cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("正在显示摄像头画面。按 'ESC' 键退出。")
    
    try:
        while True:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            # 显示画面
            cv2.imshow('Camera Feed', frame)
            
            # 按 'ESC' 键退出
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        # 释放摄像头资源并关闭窗口
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    show_camera_feed()