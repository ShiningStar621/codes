from ultralytics import YOLO
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from paddleocr import PaddleOCR
import logging
logging.getLogger("ppocr").setLevel(logging.WARNING)
##test import
def video_crop_with_ocr(
    model_path: str = r"C:\Users\qixin\runs\segment\train35\weights\best.pt",#"C:\Users\qixin\runs\segment\train33\weights\best.pt"
    video_path: str = r"Z:\对地\数据集（标靶）\2024数据集及处理代码\video_blue(0908)\新标靶9.8.MOV",  # 改为视频路径
    output_dir: str = r"C:\Users\qixin\Desktop\test-blue",
    conf_threshold: float = 0.3,
    ocr_lang: str = 'en',
    process_interval: int = 1  # 处理间隔（每N帧处理一次）
):
    """
    视频目标检测与OCR识别
    
    Args:
        process_interval: 帧处理间隔（1=处理每帧，2=每隔一帧处理）
    """
    # 初始化模型
    model = YOLO(model_path)
    ocr_engine = PaddleOCR(use_angle_cls =True, lang=ocr_lang)
    
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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{COLOR['red']}错误: 无法打开视频文件 {video_path}{COLOR['end']}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 进度条设置
    pbar = tqdm(total=total_frames, desc="Processing Video")
    
    frame_count = 0
    processed_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 按间隔处理帧
        if frame_count % process_interval == 0:
            # 执行目标检测
            results = model.predict(source=frame, conf=conf_threshold)
            
            for result in results:
                if result.boxes is None:
                    continue
                
                # 处理每个检测目标
                for obj_idx, (box, cls_id) in enumerate(zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.cls.cpu().numpy()
                )):
                    x1, y1, x2, y2 = map(int, box)
                    crop = frame[y1:y2, x1:x2]
                    cv2.imshow("cropped",crop)
                    # OCR处理
                    processed = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    processed = cv2.resize(processed, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                    ocr_result = ocr_engine.ocr(processed, cls=True)
                    
                    # 识别结果处理
                    numbers = []
                    if ocr_result[0] is not None:
                        for line in ocr_result[0]:
                            text = line[1][0].strip()
                            filtered = ''.join([c for c in text if c.isdigit()])
                            if filtered:
                                numbers.append(filtered)
                    
                    # 终端输出
                    time_stamp = frame_count / fps
                    coord_str = f"({x1},{y1})-({x2},{y2})"
                    ocr_text = ' '.join(numbers) if numbers else "未识别到数字"
                    
                    print(
                        f"{COLOR['blue']}时间: {time_stamp:.1f}s{COLOR['end']} | "
                        f"{COLOR['green']}坐标: {coord_str}{COLOR['end']} | "
                        f"{COLOR['yellow']}识别结果: {ocr_text}{COLOR['end']}"
                    )
                    with open(r"C:\Users\qixin\Desktop\test-blue.txt", "a", encoding="utf-8") as file:
                        file.write(f"识别结果: {ocr_text}"+"\n")
                    # 保存裁剪图像（按时间戳命名）
                    save_name = f"{Path(video_path).stem}_{time_stamp:.1f}s_{'_'.join(numbers) if numbers else 'NODATA'}_{obj_idx}.jpg"
                    cv2.imwrite(str(Path(output_dir) / save_name), crop)
            
            processed_count += 1
        
        frame_count += 1
        pbar.update(1)
    
    # 释放资源
    cap.release()
    pbar.close()
    print(f"\n处理完成！共处理 {processed_count}/{total_frames} 帧")

if __name__ == "__main__":
    video_crop_with_ocr(
        video_path=r"Z:\对地\数据集（标靶）\2024数据集及处理代码\video_blue(0908)\新标靶1.MOV",
        process_interval=1  # 每隔一帧处理
    )