from ultralytics import YOLO

model = YOLO(r"C:\Users\qixin\runs\segment\train35\weights\best.pt")

# 预测时直接指定保存路径
results = model.predict(
    source=r"C:\Users\qixin\Desktop\test",
    project="C:/Custom_Output",  # 指定根目录
    name="my_seg_results",       # 指定子目录名称
    exist_ok=True,               # 允许覆盖已有目录
    save=True                    # 自动保存渲染结果
)
# 遍历每张图像的检测结果
for img_idx, result in enumerate(results):
    print(f"\n=== 图像 {img_idx + 1} 检测结果 ===")
    
    # 检查是否存在检测目标
    if result.boxes is None or len(result.boxes) == 0:
        print("未检测到目标")
        continue
    
    # 打印边界框坐标（xyxy 格式）
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # 转换为 numpy 数组
    print("边界框坐标 (xyxy 格式):")
    print(boxes_xyxy)  # 输出形状为 [N, 4]，N 是检测目标数量
    
    # 打印分割掩膜信息（可选）
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()  # 掩膜张量 [N, H, W]
        print(f"分割掩膜形状: {masks.shape}")