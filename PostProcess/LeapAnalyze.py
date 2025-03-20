from LeapBase import LeapSingle
import os
import numpy as np
import cv2
import ImageCorrection

def load_image(imt_path, count=None, start_index=None, flag=None):
    data = []
    filenames = []
    if flag:
        # 获取所有图片文件
        image_files = [f for f in os.listdir(imt_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        # 从start_index开始，读取count个图片
        if start_index < len(image_files):
            end_index = min(start_index + count, len(image_files))
            image_files = image_files[start_index:end_index]
            for file in image_files:
                file_path = os.path.join(imt_path, file)
                img = cv2.imread(file_path)
                if img is not None:
                    data.append(img)
                    filenames.append(file)
    else:
        # 不使用flag时读取所有图片
        for file in os.listdir(imt_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(imt_path, file)
                img = cv2.imread(file_path)
                if img is not None:
                    data.append(img)
                    filenames.append(file)
    return data, filenames


def main():
    # model_path = "D:/EyeTrack/PostProcess/Models/LEAP071024.onnx"
    model_path = "D:/EyeTrack/PostProcess/Models/pfld-sim.onnx"
    leap = LeapSingle(model_path)
    test_img_path = "EyeTrackData/3/Left/2025-01-05_13-00-46_1505.png"
    test_img = cv2.imread(test_img_path)
    H = ImageCorrection.main()
    test_img = ImageCorrection.warpimg(test_img, H)
    test_img = test_img[300:800, 300:800]
    # 顺时针旋转60度
    height, width = test_img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), -60, 1.0)
    test_img = cv2.warpAffine(test_img, rotation_matrix, (width, height))
    # 水平翻转图片
    test_img = cv2.flip(test_img, 1)
    leap.read_image(test_img, "test_img")
    leap.run()
    mark_img = leap.draw_mark()
    cv2.imshow("mark_img", mark_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    """
    data, filenames = load_image("EyeTrackData/Eye_Test_R/")
   
    
    

    for i in range(len(data)):
        # 获取图像中心点
        height, width = data[i].shape[:2]
        center = (width // 2, height // 2)
        
        # 创建旋转矩阵，顺时针旋转30度（需要使用负角度）
        rotation_matrix = cv2.getRotationMatrix2D(center, 40, 1.0)
        
        # 执行旋转，使用黑色填充空白区域
        data[i] = cv2.warpAffine(data[i], rotation_matrix, (width, height), borderValue=(0,0,0))
    
     # 设置视频显示参数
    delay = int(1000/60)  # 每帧之间的延迟时间(毫秒)
    window_name = "Eye Tracking Results"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
        
    cv2.destroyAllWindows()
    
    
    mark_imgs = []
    for i in range(len(data)):
        leap.read_image(data[i], filenames[i])
        leap.run()
        mark_img = leap.draw_mark()
        mark_imgs.append(mark_img)

    # 设置视频显示参数
    delay = int(1000/60)  # 每帧之间的延迟时间(毫秒)
    window_name = "Eye Tracking Results"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 循环显示每一帧
    for img in mark_imgs:
        cv2.imshow(window_name, img)
        # 等待按键，如果按下'q'则退出
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    
    # 关闭所有窗口
    cv2.destroyAllWindows()
    
  
"""    
if __name__ == "__main__":
    main()

