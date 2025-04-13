import cv2
from ultralytics import YOLO

# 加载预训练模型
model = YOLO(r"C:\Users\qixin\runs\segment\train33\weights\best.pt")

# 预测并获取结果
results = model(r"C:\Users\qixin\Desktop\test")

# 提取第一个结果的检测信息
boxes = results[0].boxes.xyxy.cpu().numpy()  # 获取检测框坐标 (xyxy格式)
clss = results[0].boxes.cls.cpu().numpy()     # 类别ID
confs = results[0].boxes.conf.cpu().numpy()   # 置信度

# 读取原始图像
image = cv2.imread('input.jpg')

# 遍历每个检测目标
for idx, (box, cls, conf) in enumerate(zip(boxes, clss, confs)):
    # 转换为整数坐标
    x1, y1, x2, y2 = map(int, box)
    
    # 裁剪目标区域
    cropped = image[y1:y2, x1:x2]
    
    # 保存裁剪图像
    cv2.imwrite(f'output/crop_{idx}_{cls}_{conf:.2f}.jpg', cropped)
    
    # 打印坐标信息
    print(f'目标 {idx}: 类别={cls}, 坐标=({x1},{y1})-({x2},{y2}), 置信度={conf:.2f}')