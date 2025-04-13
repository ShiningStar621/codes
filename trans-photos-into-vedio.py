
import cv2
import os
import glob

# 参数设置
image_folder = r"C:\Users\qixin\Downloads\jsh\blue"       # 图片文件夹路径
output_folder = r"C:\Users\qixin\Desktop\test-blue"        # 指定输出文件夹路径（可以是绝对路径或相对路径）
video_name = "my_output.mp4"          # 视频文件名
fps = 20                              # 帧率
frame_size = (640, 480)               # 分辨率

# 确保输出文件夹存在（如果不存在则自动创建）
os.makedirs(output_folder, exist_ok=True)

# 拼接完整的视频保存路径
video_path = os.path.join(output_folder, video_name)  # 跨平台路径拼接

# 读取图片并生成视频
images = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

for img_path in images:
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, frame_size)
        video.write(img)
video.release()

print(f"视频已保存到：{video_path}")