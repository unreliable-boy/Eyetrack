import os
import cv2

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