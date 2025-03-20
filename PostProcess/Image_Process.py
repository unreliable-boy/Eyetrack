import cv2
import numpy as np
import os

def load_image(imt_path):
    if not os.path.exists(imt_path):
        print(f"Error: Path {imt_path} does not exist")
        exit(1)

    data = []
    # 使用opencv读取一组图片
    for file in os.listdir(imt_path):
        # 只处理常见的图片格式
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_path = os.path.join(imt_path, file)
            try:
                # 读取图片为灰度图
                img = cv2.imread(file_path)
                if img is not None:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    data.append(img)
                else:
                    print(f"Failed to load image: {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    if data:
        # 将列表转换为numpy数组
        data = np.array(data)
        print(f"Successfully loaded {len(data)} images")
        print(f"Array shape: {data.shape}")
        return data
    else:
        print("No images were loaded")

if __name__ == "__main__":

    imt_path = os.path.join("EyeTrackData", "1", "Right", "Right")
    data = load_image(imt_path)
    # 创建VideoTest文件夹（如果不存在）
    output_dir = "VideoTest"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) # 创建文件夹
    
    # 设置视频写入器
    height, width = data[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式编码
    out_path = os.path.join(output_dir, "eye_tracking.mp4")
    out = cv2.VideoWriter(out_path, fourcc, 60.0, (width, height))
    
    # 将图像序列写入视频
    for frame in data:
        out.write(frame)
    
    # 释放视频写入器
    out.release()
    print(f"Video saved to {out_path}")
    
    