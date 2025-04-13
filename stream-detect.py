from ultralytics import YOLO
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from paddleocr import PaddleOCR
import logging
from collections import deque
import numpy as np
import re
# 设置日志级别
logging.getLogger("ppocr").setLevel(logging.WARNING)
numbers=[]
# 标靶坐标
measured_points = [[0.25, 0.25, 0.75],
                   [0, -0.25, 0.75],
                   [-0.25, 0, 0.75]]
def process_ocr_result(ocr_text):
    """
    处理OCR识别结果并更新统计数组
    :param ocr_text: OCR识别出的文本
    """
    global numbers
    
    # 提取数字（匹配连续数字）
    detected_digits = re.findall(r'\d+', str(ocr_text))
    
    for digit in detected_digits:
        found = False
        
        # 遍历第一维度查找匹配项
        for i in range(len(numbers)):
            # 检查第一维度是否匹配
            if numbers[i][0] == digit:
                # 匹配则第二维度+1
                numbers[i][1] += 1
                found = True
                break
        
        # 未找到匹配项
        if not found:
            # 添加新条目（第二维度初始化为0后立即+1）
            numbers.append([digit, 0])

#############alal

# 计算两点距离
def calculate_distance(point_a, point_b):
    '''
    
    '''
    return np.linalg.norm(np.array(point_a) - np.array(point_b))

# 比较并得出目标标靶
def find_closest_point(measurements, target):
    # 计算目标点到所有测量点的距离
    distances = [np.linalg.norm(np.array(target) - np.array(point)) 
                for point in measurements]
    
    # 直接返回最小距离的索引
    return np.argmin(distances)

# 转换为相机坐标
def pixel2camera(x, y, target_depth):
    target_pixel = np.array([x, y])
    target_depth = 0.75
    camera_intrinsics = np.array([
        [241,0.0,327],
        [0.0,322,230],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]                                                                             
    ])
    fx, fy, cx, cy = camera_intrinsics[0, 0], camera_intrinsics[1, 1], camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    normalized_x = (target_pixel[0] - cx) / fx
    normalized_y = (target_pixel[1] - cy) / fy
    target_camera_coords = np.array([normalized_x * target_depth, normalized_y *target_depth, target_depth])
    return target_camera_coords                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

# 转换为世界坐标
"""def camera2enu(camera, enu, distance, pitch=90, yaw=0, roll=0):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R = R_yaw @ R_pitch @ R_roll
    T_cam_model = np.array([distance * np.cos(pitch) * np.cos(yaw),
                            distance * np.cos(pitch) * np.sin(yaw),
                            -distance * np.sin(pitch)])
    point_model = R @ np.array(camera) + T_cam_model
    enu_coord = point_model + np.array(enu)
    return enu_coord.round(4).tolist()"""

class WindowPositionManager:
    """预定义窗口位置管理类"""
    def __init__(self):
        self.positions = [(x, y) for x in range(0, 1680, 60) for y in range(20, 1000, 60)]
        self.position_queue = deque(maxlen=len(self.positions))
        
    def get_next_position(self):
        """获取下一个可用的显示位置"""
        if len(self.position_queue) < len(self.positions):
            pos = self.positions[len(self.position_queue)]
            self.position_queue.append(pos)
            return pos
        else:
            old_pos = self.position_queue.popleft()
            self.position_queue.append(old_pos)
            return old_pos

def video_crop_with_ocr(
    model_path: str = r"C:\Users\qixin\Desktop\weights\red_80eps.pt",
    video_path: str = 1,
    output_dir: str = r"C:\Users\qixin\Desktop\test-ding",
    conf_threshold: float = 0.3,
    ocr_lang: str = 'en',
    process_interval: int = 1
):
    # 初始化模型
    model = YOLO(model_path)
    ocr_engine = PaddleOCR(use_angle_cls=True, lang=ocr_lang)
    
    # 初始化窗口管理器
    window_mgr = WindowPositionManager()
    active_windows = set()  # 当前活跃窗口
    window_size = 30  # 统一窗口尺寸

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 颜色代码定义
    COLOR = {
        'blue': '\033[94m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'end': '\033[0m'
    }

    # 打开视频文件
    cap = cv2.VideoCapture(video_path,cv2.CAP_DSHOW)   #和采集卡一样1080p
    if not cap.isOpened():
        print(f"{COLOR['red']}错误: 无法打开视频文件 {video_path}{COLOR['end']}")
        return

    # 获取视频信息
    pbar = tqdm(desc="实时处理帧", unit='frame')
    
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % process_interval == 0:
                results = model.predict(source=frame, conf=conf_threshold)
                
                for result in results:
                    if result.boxes is None:
                        continue
                    
                    for obj_idx, (box, cls_id) in enumerate(zip(
                        result.boxes.xyxy.cpu().numpy(),
                        result.boxes.cls.cpu().numpy()
                    )):
                        x1, y1, x2, y2 = map(int, box)
                        enu = pixel2camera((x1+x2)/2, (y1+y2)/2 , 0.75) 
                        
                        #fenu=[0,0,1.72]
                        #enu = camera2enu(target_camera_coords,fenu,0,pitch=90, yaw=0, roll=0)
                        crop = frame[y1:y2, x1:x2]

                        # 获取显示位置并创建窗口
                        pos = window_mgr.get_next_position()
                        win_name = f"crop_{pos[0]}_{pos[1]}"
                        
                        # 调整显示尺寸并显示
                        display_img = cv2.resize(crop, (window_size, window_size))
                        cv2.imshow(win_name, display_img)
                        cv2.moveWindow(win_name, pos[0], pos[1])
                        active_windows.add(win_name)

                        # OCR处理流程
                        processed = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        processed = cv2.resize(processed, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                        ocr_result = ocr_engine.ocr(processed, cls=True)
                        if ocr_result[0] is not None:
                            for line in ocr_result :
                                if line is not None: 
                                    text = line[0][1][0]
                                    with open(r"C:\Users\qixin\Desktop\test-blue.txt", "a", encoding="utf-8") as file:
                                        file.write(f"识别结果: {line[0][1]},坐标：{enu},是{find_closest_point(measured_points,enu)+1}号标靶"+"\n") 

                                        print(enu,find_closest_point(measured_points,enu)+1)
                        
                                    process_ocr_result(text)
                        else:
                            continue
                        

                        
                # 实时更新统计显示
                #os.system('cls' if os.name == 'nt' else 'clear')

                #print("=== 实时数字统计 ===")
                
                print(numbers)
                        
            # 保持窗口响应
            if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
                break
            
            frame_count += 1
            pbar.update(1)
            
    finally:
        # 释放资源
        cap.release()
        pbar.close()

if __name__ == "__main__":
    video_crop_with_ocr(video_path=1)