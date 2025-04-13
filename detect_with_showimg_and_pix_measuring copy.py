from ultralytics import YOLO
import cv2
import os
from pathlib import Path
from paddleocr import PaddleOCR
import logging
from collections import deque, defaultdict
import numpy as np
import re
numbers = []
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
            numbers[-1][1] += 1  # 等价于初始化为1
measured_points = [[24.23, 34.67675, 0],
                   [389.34235, 23.123131, 0],
                   [456.234325, 134.31231, 0]]  # 标靶坐标

logging.getLogger("ppocr").setLevel(logging.WARNING)

def calculate_distance(point_a, point_b):
    return np.linalg.norm(np.array(point_a) - np.array(point_b))
#比较并得出目标标靶
def find_closest_point(measurements, target):
    """
    找出测量点中距离目标点最近的点
    """
    distances = [calculate_distance(target, point) for point in measurements]
    
    return np.argmin(distances)
# 转换为相机坐标
def pixel2camera(x, y , depth):
    #目标的坐标值

    target_pixel = np.array([x, y])  

    #深度值
    target_depth = depth  

    #相机内参
    '''
    fx  0  cx
    0   fy cy
    0   0  1
    使用了3.25标定的数据
    '''
    camera_intrinsics = np.array([
        [9.65464811e+02, 0.00000000e+00, 1.24168475e+03],
        [0.00000000e+00, 9.62873088e+02, 7.08478371e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    fx, fy, cx, cy = camera_intrinsics[0, 0], camera_intrinsics[1, 1], camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    normalized_x = (target_pixel[0] - cx) / fx
    normalized_y = (target_pixel[1] - cy) / fy

    target_camera_coords = np.array([normalized_x * target_depth, normalized_y * target_depth, target_depth])

    return target_camera_coords
# 转换为世界坐标（缺乏飞机坐标的输入）
def camera2enu(camera, enu, distance, pitch=90, yaw=0, roll=0):
    """
    camera: 相机坐标系的坐标
    enu: ENU坐标
    distance: 相机与航模之间的直线距离（m）
    pitch: 俯仰角（弧度，向下为正，默认90向正下方）
    yaw: 偏航角（弧度，默认0）
    roll: 滚转角（弧度，默认0）
    """
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)
    
    R_yaw = np.array([[cy, -sy, 0],
                      [sy, cy, 0],
                      [0, 0, 1]])
    
    R_pitch = np.array([[cp, 0, sp],
                        [0, 1, 0],
                        [-sp, 0, cp]])
    
    R_roll = np.array([[1, 0, 0],
                       [0, cr, -sr],
                       [0, sr, cr]])
    
    R = R_yaw @ R_pitch @ R_roll
    

    T_cam_model = np.array([distance * np.cos(pitch) * np.cos(yaw),
                            distance * np.cos(pitch) * np.sin(yaw),
                            -distance * np.sin(pitch)])
    

    point_model = R @ np.array(camera) + T_cam_model
    
    enu_coord = point_model + np.array(enu)
    
    return enu_coord.round(4).tolist()

class WindowPositionManager:
    """预定义窗口位置管理类"""
    def __init__(self):
        # 预定义10个不重叠的显示位置 (x, y)
        self.positions = [
            (0,20),
            (0,80),
            (0,140),
            (0,200),
            (0,260),
            (0,320),
            (0,380),
            (0,440),
            (0,500),
            (0,560),
            (0,620),
            (0,680),
            (0,740),
            (0,800),
            (0,860),
            (0,920),
            (0,980),
            (60,20),
            (60,80),
            (60,140),
            (60,200),
            (60,260),
            (60,320),
            (60,380),
            (60,440),
            (60,500),
            (60,560),
            (60,620),
            (60,680),
            (60,740),
            (60,800),
            (60,860),
            (60,920),
            (60,980),
            (120,20),
            (120,80),
            (120,140),
            (120,200),
            (120,260),
            (120,320),
            (120,380),
            (120,440),
            (120,500),
            (120,560),
            (120,620),
            (120,680),
            (120,740),
            (120,800),
            (120,860),
            (120,920),
            (120,980),
            (180,20),
            (180,80),
            (180,140),
            (180,200),
            (180,260),
            (180,320),
            (180,380),
            (180,440),
            (180,500),
            (180,560),
            (180,620),
            (180,680),
            (180,740),
            (180,800),
            (180,860),
            (180,920),
            (180,980),
            (240,20),
            (240,80),
            (240,140),
            (240,200),
            (240,260),
            (240,320),
            (240,380),
            (240,440),
            (240,500),
            (240,560),
            (240,620),
            (240,680),
            (240,740),
            (240,800),
            (240,860),
            (240,920),
            (240,980),
            (300,20),
            (300,80),
            (300,140),
            (300,200),
            (300,260),
            (300,320),
            (300,380),
            (300,440),
            (300,500),
            (300,560),
            (300,620),
            (300,680),
            (300,740),
            (300,800),
            (300,860),
            (300,920),
            (300,980),
            (360,20),
            (360,80),
            (360,140),
            (360,200),
            (360,260),
            (360,320),
            (360,380),
            (360,440),
            (360,500),
            (360,560),
            (360,620),
            (360,680),
            (360,740),
            (360,800),
            (360,860),
            (360,920),
            (360,980),
            (420,20),
            (420,80),
            (420,140),
            (420,200),
            (420,260),
            (420,320),
            (420,380),
            (420,440),
            (420,500),
            (420,560),
            (420,620),
            (420,680),
            (420,740),
            (420,800),
            (420,860),
            (420,920),
            (420,980),
            (480,20),
            (480,80),
            (480,140),
            (480,200),
            (480,260),
            (480,320),
            (480,380),
            (480,440),
            (480,500),
            (480,560),
            (480,620),
            (480,680),
            (480,740),
            (480,800),
            (480,860),
            (480,920),
            (480,980),
            (540,20),
            (540,80),
            (540,140),
            (540,200),
            (540,260),
            (540,320),
            (540,380),
            (540,440),
            (540,500),
            (540,560),
            (540,620),
            (540,680),
            (540,740),
            (540,800),
            (540,860),
            (540,920),
            (540,980),
            (600,20),  
            (660,20),  
            (720,20),  
            (780,20),  
            (840,20),  
            (900,20),  
            (960,20),  
            (600,80),  
            (660,80),  
            (720,80),  
            (780,80),  
            (840,80),  
            (900,80),  
            (960,80),  
            (600,140),  
            (660,140),  
            (720,140),  
            (780,140),  
            (840,140),  
            (900,140),  
            (960,140),  
            (600,200),  
            (660,200),  
            (720,200),  
            (780,200),  
            (840,200),  
            (900,200),  
            (960,200),  
            (600,260),  
            (660,260),  
            (720,260),  
            (780,260),  
            (840,260),  
            (900,260),  
            (960,260),  
            (600,320),  
            (660,320),  
            (720,320),  
            (780,320),  
            (840,320),  
            (900,320),  
            (960,320),  
            (600,380),  
            (660,380),  
            (720,380),  
            (780,380),  
            (840,380),  
            (900,380),  
            (960,380),  
            (600,440),  
            (660,440),  
            (720,440),  
            (780,440),  
            (840,440),  
            (900,440),  
            (960,440),  
            (600,500),  
            (660,500),  
            (720,500),  
            (780,500),  
            (840,500),  
            (900,500),  
            (960,500),  
            (600,560),  
            (660,560),  
            (720,560),  
            (780,560),  
            (840,560),  
            (900,560),  
            (960,560),  
            (600,620),  
            (660,620),  
            (720,620),  
            (780,620),  
            (840,620),  
            (900,620),  
            (960,620),  
            (600,680),  
            (660,680),  
            (720,680),  
            (780,680),  
            (840,680),  
            (900,680),  
            (960,680),  
            (600,740),  
            (660,740),  
            (720,740),  
            (780,740),  
            (840,740),  
            (900,740),  
            (960,740),  
            (600,800),  
            (660,800),  
            (720,800),  
            (780,800),  
            (840,800),  
            (900,800),  
            (960,800),  
            (600,860),  
            (660,860),  
            (720,860),  
            (780,860),  
            (840,860),  
            (900,860),  
            (960,860),  
            (600,920),  
            (660,920),  
            (720,920),  
            (780,920),  
            (840,920),  
            (900,920),  
            (960,920),  
            (600,980),  
            (660,980),  
            (720,980),  
            (780,980),  
            (840,980),  
            (900,980),  
            (960,980),
            (1020,20),	(1080,20),	(1140,20),	(1200,20),	(1260,20),	(1320,20),	(1380,20),
            (1020,80),	(1080,80),	(1140,80),	(1200,80),	(1260,80),	(1320,80),	(1380,80),
            (1020,140),	(1080,140),	(1140,140),	(1200,140),	(1260,140),	(1320,140),	(1380,140),
            (1020,200),	(1080,200),	(1140,200),	(1200,200),	(1260,200),	(1320,200),	(1380,200),
            (1020,260),	(1080,260),	(1140,260),	(1200,260),	(1260,260),	(1320,260),	(1380,260),
            (1020,320),	(1080,320),	(1140,320),	(1200,320),	(1260,320),	(1320,320),	(1380,320),
            (1020,380),	(1080,380),	(1140,380),	(1200,380),	(1260,380),	(1320,380),	(1380,380),
            (1020,440),	(1080,440),	(1140,440),	(1200,440),	(1260,440),	(1320,440),	(1380,440),
            (1020,500),	(1080,500),	(1140,500),	(1200,500),	(1260,500),	(1320,500),	(1380,500),
            (1020,560),	(1080,560),	(1140,560),	(1200,560),	(1260,560),	(1320,560),	(1380,560),
            (1020,620),	(1080,620),	(1140,620),	(1200,620),	(1260,620),	(1320,620),	(1380,620),
            (1020,680),	(1080,680),	(1140,680),	(1200,680),	(1260,680),	(1320,680),	(1380,680),
            (1020,740),	(1080,740),	(1140,740),	(1200,740),	(1260,740),	(1320,740),	(1380,740),
            (1020,800),	(1080,800),	(1140,800),	(1200,800),	(1260,800),	(1320,800),	(1380,800),
            (1020,860),	(1080,860),	(1140,860),	(1200,860),	(1260,860),	(1320,860),	(1380,860),
            (1020,920),	(1080,920),	(1140,920),	(1200,920),	(1260,920),	(1320,920),	(1380,920),
            (1020,980),	(1080,980),	(1140,980),	(1200,980),	(1260,980),	(1320,980),	(1380,980),
            (1440,20),	(1500,20),	(1560,20),
            (1440,80),	(1500,80),	(1560,80),
            (1440,140),	(1500,140),	(1560,140),
            (1440,200),	(1500,200),	(1560,200),
            (1440,260),	(1500,260),	(1560,260),
            (1440,320),	(1500,320),	(1560,320),
            (1440,380),	(1500,380),	(1560,380),
            (1440,440),	(1500,440),	(1560,440),
            (1440,500),	(1500,500),	(1560,500),
            (1440,560),	(1500,560),	(1560,560),
            (1440,620),	(1500,620),	(1560,620),
            (1440,680),	(1500,680),	(1560,680),
            (1440,740),	(1500,740),	(1560,740),
            (1440,800),	(1500,800),	(1560,800),
            (1440,860),	(1500,860),	(1560,860),
            (1440,920),	(1500,920),	(1560,920),
            (1440,980),	(1500,980),	(1560,980),
            (1620,20),
            (1620,80),
            (1620,140),
            (1620,200),
            (1620,260),
            (1620,320),
            (1620,380),
            (1620,440),
            (1620,500),
            (1620,560),
            (1620,620),
            (1620,680),
            (1620,740),
            (1620,800),
            (1620,860),
            (1620,920),
            (1620,980),



    

            


        ]
        self.position_queue = deque(maxlen=len(self.positions))
    def get_next_position(self):
        """获取下一个可用的显示位置"""
        if len(self.position_queue) < len(self.positions):
            pos = self.positions[len(self.position_queue)]
            self.position_queue.append(pos)
            return pos
        else:
            # 循环使用最早的位置
            old_pos = self.position_queue.popleft()
            self.position_queue.append(old_pos)
            return old_pos

def video_crop_with_ocr(
    model_path: str = r"C:\Users\qixin\Desktop\weights\red_80eps.pt",
    video_path: str = r"Z:\对地\数据集（标靶）\2024数据集及处理代码\video_blue(0908)\新标靶9.8.MOV",
    output_dir: str = r"C:\Users\qixin\Desktop\test-blue",
    conf_threshold: float = 0.5,
    ocr_lang: str = 'en',
    process_interval: int = 1
):
    model = YOLO(model_path)
    ocr_engine = PaddleOCR(use_angle_cls=True, lang=ocr_lang,conf_threshold=0.95)
    window_mgr = WindowPositionManager()
    active_windows = set()
    window_size = 30
    digit_counts = defaultdict(int)  # 新增：数字统计字典

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("\033[91m错误: 无法打开视频文件\033[0m")
        return

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

                    for box, cls_id in zip(result.boxes.xyxy.cpu().numpy(),
                                         result.boxes.cls.cpu().numpy()):
                        x1, y1, x2, y2 = map(int, box)
                        crop = frame[y1:y2, x1:x2]

                        # 窗口显示逻辑保持不变...
                        pos = window_mgr.get_next_position()
                        win_name = f"crop_{pos[0]}_{pos[1]}"
                        display_img = cv2.resize(crop, (window_size, window_size))
                        cv2.imshow(win_name, display_img)
                        cv2.moveWindow(win_name, pos[0], pos[1])
                        active_windows.add(win_name)

                        # OCR处理
                        processed = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        processed = cv2.resize(processed, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                        ocr_result = ocr_engine.ocr(processed, cls=True)
                        if ocr_result[0] is not None:
                            for line in ocr_result :
                                if line is not None: 
                                    text = line[0][1][0]
                                    with open(r"C:\Users\qixin\Desktop\test-blue.txt", "a", encoding="utf-8") as file:
                                        file.write(f"识别结果: {line[0][1]}"+"\n") 
                        
                                    process_ocr_result(text)
                        else:
                            continue
                        

                        
                # 实时更新统计显示
                os.system('cls' if os.name == 'nt' else 'clear')
                print("=== 实时数字统计 ===")
                print(numbers)
                """for num, count in sorted(digit_counts.items(), key=lambda x: int(x[0])):
                    print(f"数字 {num}: 出现 {count} 次")
"""
            if cv2.waitKey(1) & 0xFF == 27:
                break
            frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n=== 最终统计结果 ===")
        for num, count in sorted(digit_counts.items(), key=lambda x: int(x[0])):
            print(f"数字 {num}: 出现 {count} 次")

if __name__ == "__main__":
    video_crop_with_ocr(
        video_path=r"Z:\对地\数据集（标靶）\飞机端视频\DJI_0009.MP4",
        process_interval=5
    )