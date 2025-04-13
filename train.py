# train.py
from ultralytics import YOLO

def train_model():
    model = YOLO(r"C:\Users\qixin\runs\segment\train46\weights\best.pt")  # 或加载预训练模型 YOLO("yolov8n-seg.pt")
    results = model.train(
        data="C:/Users/qixin/Desktop/p/ultralytics-main/ultralytics/models/ngyfdd/train_vs_two_targets.yaml",
        epochs=80,
        imgsz=640,
    )

if __name__ == '__main__':
    train_model()