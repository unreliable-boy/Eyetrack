import os
import onnxruntime
# import onnx
import numpy as np
import cv2
import time
import math
from queue import Queue
import threading

# from utils.misc_utils import resource_path
from pathlib import Path


"""
眨眼检测
"""

os.environ["OMP_NUM_THREADS"] = "1"

frames = 0
models = Path("Models")


def run_model(input_queue, output_queue, session):
    while True:
        frame = input_queue.get()
        if frame is None:
            break

        img_np = np.array(frame, dtype=np.float32) / 255.0
        gray_img = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]

        gray_img = np.expand_dims(np.expand_dims(gray_img, axis=0), axis=0)

        ort_inputs = {session.get_inputs()[0].name: gray_img}
        pre_landmark = session.run(None, ort_inputs)
        pre_landmark = np.reshape(pre_landmark, (-1, 2))
        output_queue.put((frame, pre_landmark))



def image_process(frame):
        frame = cv2.resize(frame, (112, 112))
        img_np = np.array(frame, dtype=np.float32) / 255.0  #归一
        gray_img = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2] #3RGB值
        
        gray_img = np.expand_dims(np.expand_dims(gray_img, axis=0), axis=0)
    
        return gray_img



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
    
        
class LeapSingle():
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = onnxruntime.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.img_name = ""
        self.img = None
        
        self.draw_color =[
            (0, 0, 255),     # 红色
            (0, 255, 0),     # 绿色 
            (255, 0, 0),     # 蓝色
            (255, 255, 0),   # 青色
            (255, 0, 255),   # 洋红色
            (0, 255, 255),   # 黄色
            (255, 255, 255), # EYE Center Color
            (0, 128, 0),     # 深绿色
            (0, 0, 0),       # 白色
            (128, 128, 0),   # 橄榄色
            (128, 0, 128),   # 紫色
            (0, 128, 128)    # 青绿色
        ]
    
    def read_image(self,img, img_name):
        self.img = img
        self.img_name = img_name
       
    
    def img_process(self, frame):
        frame = cv2.resize(frame, (112, 112))
        img_np = np.array(frame, dtype=np.float32) / 255.0  #归一
        if len(img_np.shape) == 3:
            gray_img = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2] #3RGB值
        else:
            gray_img = img_np
        
        gray_img = np.expand_dims(np.expand_dims(gray_img, axis=0), axis=0)
        self.gray_img = gray_img
        
    def run(self):
        if self.img is not None:
            self.img_process(self.img)
            Mark = self.session.run(None, {self.input_name: self.gray_img})
            Mark = np.reshape(Mark, (-1, 2))
            self.Mark = Mark
        
    
    def get_mark(self):
        return self.Mark
    
    def draw_mark(self):
        times_ = 0
        img = self.img.copy()
        for points in self.Mark:
            x, y = points
            x = int(x * 224)
            y = int(y * 224)
            color = self.draw_color[times_]
            cv2.circle(img, (x, y), 1, (color), -1)
            cv2.circle(img, (x, y), 3, (color), -1)
            times_ += 1
        return img
    
    


if __name__ == "__main__":
    # model_path = "D:/EyeTrack_Break/PostProcess/Models/LEAP062120246epoch.onnx"
    model_path = "D:/EyeTrack/PostProcess/Models/LEAP071024.onnx"
    # 使用onnxruntime 加载模型
    session = onnxruntime.InferenceSession(model_path)
    
    img_Test = cv2.imread("EyeTrackData/1/Left/2024-12-30_18-16-20_3799.png")
    
    #旋转图像 逆时针90度
    # 获取图像中心点
    height, width = img_Test.shape[:2]
    center = (width // 2, height // 2)
    
    # 创建旋转矩阵，45度，1.0表示保持原始比例
    rotation_matrix = cv2.getRotationMatrix2D(center, -50, 1.0)
    
    # 执行旋转，borderValue指定填充颜色为黑色
    img_Test = cv2.warpAffine(img_Test, rotation_matrix, (width, height), borderValue=(0,0,0))
    # img_Test = cv2.rotate(img_Test, cv2.ROTATE_90_CLOCKWISE)
    
    
    # 保存原始图像用于可视化
    img_vis = cv2.resize(img_Test.copy(), (224, 224))

    
    # 处理图像
    img_Test = image_process(img_Test)
    
    # 获取模型输入名称
    input_name = session.get_inputs()[0].name
    # 创建输入字典
    ort_inputs = {input_name: img_Test}
    
    # 运行推理
    test_mark = session.run(None, ort_inputs)
    test_mark = np.reshape(test_mark, (-1, 2))
    print(test_mark)

    img_vis = cv2.resize(img_vis, (224, 224))
    # 可视化并放大图像
    time_ = 0
    for x,y in test_mark:
        x = int(x * 224)
        y = int(y * 224)
        cv2.circle(img_vis, (x, y), 1, (0, 0, 255), -1)
        cv2.circle(img_vis, (x, y), 3, (255, 255, 0), -1)
    
    
    
    cv2.imshow("img", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    

    
   #  cv2.imwrite("img_vis.png", img_vis)
    
    