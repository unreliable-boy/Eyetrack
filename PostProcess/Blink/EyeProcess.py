"""
提取瞳孔点后规划眼睛区域喂给model    
"""
from collections import OrderedDict
import cv2
import numpy as np
import os
import imutils 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import LeapBase
from tensorflow.python.keras.models import load_model
import tensorflow as tf


BLINK_DETECT_BASE_SIZE = (17, 13)   #17:13

path = "D:\EyeTrack\PostProcess\Models\Haar\haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(path)  

# image = cv2.imread("./EyeTrackData/3/Left/2025-01-05_13-00-46_1505.png")
image = cv2.imread("./EyeTrackData/XXL_0114_Readtxt_Dark/Left/2025-01-14_22-59-28_1753.png")
image = cv2.flip(image, 1)  # 1表示水平翻转


# 全局变量，用于存储点击的点
points = []
gray = None

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Image', image)
        if len(points) == 2:
            # cv2.line(image, points[0], points[1], (0, 0, 255), 2)
            cv2.imshow('Image', image)
            rotate_image(image, points[0], points[1])

def rotate_image(image, point1, point2):
    global gray
    # 计算两点之间的角度
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    # 计算线段与水平线的夹角
    angle = np.degrees(np.arctan2(dy, dx))
    
    # 获取图像中心点
    center = (image.shape[1] // 2, image.shape[0] // 2)
    
    # 计算旋转矩阵 - 注意这里不需要取负
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 旋转图像
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    gray = rotated_image.copy()
    
    # 计算旋转后的点的新位置
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([np.array(points), ones])
    transformed_points = M.dot(points_ones.T).T
    
    
    # 在旋转后的图像上画出点和线
    for point in transformed_points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(rotated_image, (x, y), 5, (0, 255, 0), -1)
    
    cv2.line(rotated_image, 
             (int(transformed_points[0][0]), int(transformed_points[0][1])),
             (int(transformed_points[1][0]), int(transformed_points[1][1])),
             (0, 0, 255), 2)
    
    # 显示旋转后的图像
    cv2.imshow('Rotated Image', rotated_image)
    
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', click_event)
# 显示图像
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
net_img = gray.copy()
eye_centers = []
eye_rects = []
eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for eye in eyes:
    eye_center = (eye[0] + eye[2] // 2, eye[1] + eye[3] // 2)
    eye_centers.append(eye_center)
    eye_rect = (eye[0], eye[1], eye[2], eye[3])

# 如果检测到多个眼睛，选择距离图像中心最近的一个
if len(eye_centers) > 1:
    image_center = (gray.shape[1] // 2, gray.shape[0] // 2)
    min_dist = float('inf')
    closest_center = None
    
    for i, center in enumerate(eye_centers):
        dist = ((center[0] - image_center[0])**2 + (center[1] - image_center[1])**2)**0.5
        if dist < min_dist:
            min_dist = dist
            closest_center = center
            closest_rect = eye_rects[i]  # 记录对应的rect
    eye_rects = [closest_rect]  # 更新eye_rects列表只保留最近的rect
    eye_centers = [closest_center]

# 在灰度图像上绘制眼睛中心点
if len(eye_centers) > 0:
    center = eye_centers[0]
    gray_show = gray.copy()
    cv2.circle(gray_show, center, 5, (255, 255, 255), -1)  # 在灰度图上用白色圆点标记
    
    # 计算截取区域的边界
    half_width = BLINK_DETECT_BASE_SIZE[0] *4
    half_height = BLINK_DETECT_BASE_SIZE[1] *4
    
    # 确保截取区域不超出图像边界
    x1 = max(0, center[0] - half_width)
    x2 = min(gray.shape[1], center[0] + half_width)
    y1 = max(0, center[1] - half_height)
    y2 = min(gray.shape[0], center[1] + half_height)
    
    # 截取眼睛区域
    eye_roi = gray[y1:y2, x1:x2]
    eye_roi_show = gray_show[y1:y2, x1:x2]
    # 调整到指定大小
    # eye_roi = cv2.resize(eye_roi, (BLINK_DETECT_BASE_SIZE[0]*2, BLINK_DETECT_BASE_SIZE[1]*2))
    

# 检查模型文件是否存在
model_path = 'D:/EyeTrack/PostProcess/Models/2018_12_17_22_58_35.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

try:
    model = load_model(model_path)
    model.summary()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    sys.exit(1)


eye_img = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
IMG_SIZE = (34, 26)

# 确保图像大小正确
eye_img = cv2.resize(eye_img, (IMG_SIZE[0], IMG_SIZE[1]))
# 正确处理输入图像
eye_input = eye_img.reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.0
# 检查图像维度
print(f"Eye image shape before reshape: {eye_input.shape}")
# 检查处理后的维度
print(f"Eye input shape after reshape: {eye_input.shape}")

eye_input = tf.convert_to_tensor(eye_input)
    
    # 使用模型进行预测
pred = model(eye_input, training=False)
pred = pred.numpy()

print(f"Prediction: {pred}")

cv2.imshow("Eye ROI", eye_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
