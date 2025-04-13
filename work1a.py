import cv2
import numpy as np
import glob

# 棋盘格的行数和列数
chessboard_size = (11, 8)  # 棋盘格每个格子的实际大小（单位：毫米）
square_size = 15

# 用于存储角点的坐标
obj_points = []  # 3D点
img_points = []  # 2D点

# 生成棋盘格的3D点坐标
obj_p = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
obj_p[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# 读取图像文件
images = glob.glob(r'C:\Users\qixin\Desktop\result\2/*.jpg')  # 修改为你的图像文件路径

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCornersSB(img,chessboard_size,None)
    if ret:
        obj_points.append(obj_p)
        img_points.append(corners)
        print(f"成功检测到棋盘格角点：{fname}")
        # 绘制角点
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 相机标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

# 打印结果
print("相机内参矩阵:")
print(camera_matrix)
print("畸变系数:")
print(dist_coeffs)
