import os
import cv2

def process_images(folder_path, output_file):
    # 确保输出文件不存在（避免覆盖）
    if os.path.exists(output_file):
        os.remove(output_file)

    # 遍历文件夹中的所有图片
    for filename in os.listdir(folder_path):
        # 检查文件扩展名是否为图片格式
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 构造图片的完整路径
            image_path = os.path.join(folder_path, filename)
            
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图片：{image_path}")
                continue
            
            # 显示图片
            cv2.imshow("Image", image)
            cv2.waitKey(1)  # 确保图片窗口显示
            
            # 等待键盘输入
            print(f"正在处理图片：{image_path}")
            print("请输入一个数字，或按空格键表示图片内容为空。")
            while True:
                key = cv2.waitKey(0) & 0xFF  # 等待按键输入
                if key == ord(' '):  # 按空格键
                    num = "None"
                    break
                elif key == ord('\r'):  # 按回车键
                    try:
                        num = input("请输入一个数字: ")
                        if num.strip() == "":
                            print("输入无效，请输入一个数字或按空格键。")
                            continue
                        num = int(num)
                    except ValueError:
                        print("输入无效，请输入一个数字。")
                        continue
                    break
                else:
                    print("无效按键，请按空格键或回车键。")
            
            # 将结果写入输出文件
            with open(output_file, 'a') as f:
                f.write(f"{image_path} {num}\n")
            
            # 关闭图片窗口
            cv2.destroyAllWindows()

    print("处理完成。结果已保存到文件：", output_file)

if __name__ == "__main__":

    # 调用函数处理图片
    process_images(r"C:\Users\qixin\Desktop\testtrain", r"C:\Users\qixin\Desktop\train-ocr\rec\rec_gt_train.txt")