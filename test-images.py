from ultralytics import YOLO
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from paddleocr import PaddleOCR
import logging
import glob

logging.getLogger("ppocr").setLevel(logging.WARNING)

def image_folder_crop_with_ocr(
    model_path: str = r"C:\Users\qixin\runs\segment\train35\weights\best.pt",
    image_folder: str = r"C:\Users\qixin\Downloads\jsh\blue",  # 图片文件夹路径
    output_dir: str = r"C:\Users\qixin\Desktop\test-blue",
    conf_threshold: float = 0.3,
    ocr_lang: str = 'en',
    process_interval: int = 1
):
    """
    图片文件夹目标检测与OCR识别
    
    Args:
        process_interval: 图片处理间隔（1=处理每张，2=每隔一张处理）
    """
    # 初始化模型
    model = YOLO(model_path)
    ocr_engine = PaddleOCR(use_angle_cls=True, lang=ocr_lang)
    
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
    
    # 获取所有图片文件（支持多种格式）
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    
    # 自然排序确保正确顺序
    
    if not image_files:
        print(f"{COLOR['red']}错误: 未找到图片文件{COLOR['end']}")
        return
    
    # 进度条设置
    pbar = tqdm(total=len(image_files), desc="Processing Images")
    
    processed_count = 0
    
    for idx, img_path in enumerate(image_files):
        # 按间隔处理图片
        if idx % process_interval != 0:
            continue
            
        # 读取图片
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"{COLOR['red']}警告: 无法读取图片 {img_path}{COLOR['end']}")
            continue
        
        # 获取基础文件名
        base_name = Path(img_path).stem
        
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
                coord_str = f"({x1},{y1})-({x2},{y2})"
                ocr_text = ' '.join(numbers) if numbers else "未识别到数字"
                
                print(
                    f"{COLOR['blue']}图片: {base_name}{COLOR['end']} | "
                    f"{COLOR['green']}坐标: {coord_str}{COLOR['end']} | "
                    f"{COLOR['yellow']}识别结果: {ocr_text}{COLOR['end']}"
                )
                
                # 保存结果到文本文件
                with open(Path(output_dir) / "results.txt", "a", encoding="utf-8") as file:
                    file.write(f"图片: {base_name} | 坐标: {coord_str} | 识别结果: {ocr_text}\n")
                
                # 保存裁剪图像（使用原图文件名+编号）
                save_name = f"{base_name}_{'_'.join(numbers) if numbers else 'NODATA'}_{obj_idx}.jpg"
                cv2.imwrite(str(Path(output_dir) / save_name), crop)
        
        processed_count += 1
        pbar.update(1)
    
    pbar.close()
    print(f"\n处理完成！共处理 {processed_count}/{len(image_files)} 张图片")

if __name__ == "__main__":
    image_folder_crop_with_ocr(
        image_folder=r"C:\Users\qixin\Downloads\jsh\blue",  # 修改为你的图片文件夹路径
        process_interval=1
    )