import os
import cv2

def process_images(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    if not image_files:
        print("文件夹中没有找到图片文件！")
        return

    output_file = os.path.join(folder_path, "image_labels.txt")
    results = []

    for idx, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue

        window_name = f"Image Labeling ({idx+1}/{len(image_files)})"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        label = None
        display_img = img.copy()
        
        while True:
            # 显示当前图片和提示信息
            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(0)
            
            # ESC 键退出程序
            if key == 27:  
                cv2.destroyAllWindows()
                print("用户中断操作")
                return
            
            # 空格键标记为空
            if key == 32:  
                label = " "
                break
            
            # 输入第一个数字
            if 48 <= key <= 57 and label is None:
                first_digit = chr(key)
                # 在图片上显示已输入的第一个数字
                display_img = img.copy()
                cv2.putText(display_img, f"First Digit: {first_digit}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_img, "Press second digit...", (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow(window_name, display_img)
                
                # 等待第二个数字
                second_key = cv2.waitKey(0)
                if 48 <= second_key <= 57:
                    second_digit = chr(second_key)
                    label = first_digit + second_digit
                    break
                elif second_key == 27:  # 第二个输入按ESC取消
                    display_img = img.copy()
                    continue
                else:  # 非数字输入重置
                    display_img = img.copy()
            
        # 记录结果
        results.append(f"{img_path}\t{label}")
        cv2.destroyWindow(window_name)
        print(f"已处理: {img_path} -> {label}")

    # 保存结果到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))
    print(f"标注完成！结果已保存到: {output_file}")


if __name__ == "__main__":
    folder = r"C:\Users\qixin\Desktop\test-blue"
    
    if os.path.isdir(folder):
        process_images(folder)
    else:
        print("无效的文件夹路径！")