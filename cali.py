import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from pathlib import Path

def calibrate_camera(images_path, chessboard_size=(11, 8), save_results=True):
    """
    相机标定函数
    
    参数:
        images_path (str): 包含标定图像的目录路径或通配符路径
        chessboard_size (tuple): 棋盘格内角点数量 (columns, rows)
        save_results (bool): 是否保存标定结果
        
    返回:
        ret: 标定误差
        mtx: 相机内参矩阵
        dist: 畸变系数
        rvecs: 旋转向量
        tvecs: 平移向量
    """
    # 准备对象点 (0,0,0), (1,0,0), (2,0,0) ..., (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    
    # 存储对象点和图像点的数组
    objpoints = []  # 真实世界中的3D点
    imgpoints = []  # 图像中的2D点
    
    # 获取标定图像路径
    images = glob.glob(images_path)
    if not images:
        raise ValueError(f"未找到图像文件: {images_path}")
    
    # 遍历所有图像
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        # 如果找到，添加对象点和图像点
        if ret:
            objpoints.append(objp)
            
            # 提高角点检测精度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # 绘制并显示角点
            img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('Chessboard Corners', img)
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    
    if not objpoints:
        raise ValueError("未在任何图像中找到棋盘格角点，请检查棋盘格大小和图像质量")
    
    # 进行相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # 评估标定误差
    mean_error = evaluate_calibration(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    print(f"标定完成，平均重投影误差: {mean_error:.3f} 像素")
    
    # 保存标定结果
    if save_results:
        save_calibration_results(mtx, dist, mean_error)
    
    return ret, mtx, dist, rvecs, tvecs

def evaluate_calibration(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    """评估标定质量，计算平均重投影误差"""
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    return total_error / len(objpoints)

def save_calibration_results(mtx, dist, error):
    """保存标定结果到文件"""
    calib_data = {
        'camera_matrix': mtx.tolist(),
        'distortion_coefficients': dist.tolist(),
        'reprojection_error': error
    }
    
    import json
    with open('camera_calibration.json', 'w') as f:
        json.dump(calib_data, f, indent=4)
    print("标定结果已保存到 camera_calibration.json")

def undistort_image(image_path, mtx, dist):
    """使用标定结果校正图像畸变"""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # 优化相机矩阵
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    # 校正畸变
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # 裁剪图像
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    # 显示结果
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('原始图像')
    plt.subplot(122), plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)), plt.title('校正后图像')
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 标定相机
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(
        images_path="C:/Users/qixin/Desktop/result/2/*.jpg",  # 替换为你的标定图像路径
        chessboard_size=(11, 8))  # 根据你的棋盘格调整
    
    # 使用标定结果校正图像
    test_image = "C:/Users/qixin/Desktop/result/1/frame_93.jpg"  # 替换为测试图像路径
    if Path(test_image).exists():
        undistort_image(test_image, mtx, dist)
    else:
        print(f"测试图像 {test_image} 不存在，跳过校正演示")