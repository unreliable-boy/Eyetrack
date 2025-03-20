import os
import cv2
import numpy as np
from imutils import face_utils
from tensorflow.python.keras.models import load_model
import tensorflow as tf
import pandas as pd  # 导入pandas库
import re
from tqdm import tqdm

IMG_SIZE = (34, 26)
haar_path = "PostProcess/Models/Haar/haarcascade_eye.xml"


# ------------------------- 第一部分：图像处理和眨眼检测 -------------------------

def load_image(img_path, count=None, start_index=None, flag=None):
    data = []
    filenames = []

    def get_frame_number(filename):
        name_part = os.path.splitext(filename)[0]
        match = re.search(r'_(\d+)$', name_part)  # 匹配末尾的连续数字
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"文件名 '{filename}' 的帧号格式错误")

    # 获取并排序所有文件
    image_files = [
        f for f in os.listdir(img_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]
    image_files.sort(key=lambda x: get_frame_number(x))  # 按数值排序

    # 截取指定范围的图片
    if flag:
        if start_index is None or count is None:
            raise ValueError("start_index 和 count 需同时提供")
        start_index = max(0, start_index)
        end_index = min(start_index + count, len(image_files))
        selected_files = image_files[start_index:end_index]
    else:
        selected_files = image_files

    # 加载图片
    for file in selected_files:
        file_path = os.path.join(img_path, file)
        img = cv2.imread(file_path)
        if img is not None:
            data.append(img)
            filenames.append(file)

    return data, filenames

def rotate_first_img(img):
    points = []  # 存储选择的点
    M = None  # 初始化旋转矩阵

    def select_point(event, x, y, flags, param):
        nonlocal M  # 使用外部作用域的M
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((x, y))
                cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
                cv2.imshow('image', img)
                if len(points) == 2:
                    # 计算旋转角度
                    dx = points[1][0] - points[0][0]
                    dy = points[1][1] - points[0][1]
                    angle = np.arctan2(dy, dx) * 180 / np.pi  # 计算角度
                    center = (points[0][0], points[0][1])  # 旋转中心为第一个点
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # 计算旋转矩阵
                    # rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))  # 旋转图像
                    # cv2.imshow('Rotated Image', rotated_img)  # 显示旋转后的图像

    cv2.imshow('image', img)
    cv2.setMouseCallback('image', select_point)  # 设置鼠标回调
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return M  # 返回旋转矩阵


def detect_pupil(img):
    # 使用Haar级联分类器检测眼睛
    eye_cascade = cv2.CascadeClassifier(haar_path)
    eyes = eye_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    # 如果检测到眼睛，返回眼睛的中心坐标
    if len(eyes) > 0:
        for (ex, ey, ew, eh) in eyes:
            cv2.circle(img, (ex + ew // 2, ey + eh // 2), 5, (0, 0, 255), -1)
            ptA = (ex, ey)
            ptB = (ex + ew, ey + eh)
            # cv2.imshow('image', img)
            # cv2.waitKey(1)
            # print ("检测到瞳孔位置.")
        return eyes[0][0] + eyes[0][2] // 2, eyes[0][1] + eyes[0][3] // 2

    else:
        # print("没有检测到位置，需要结束进程重试.")
        return None
    
def detect_pupil_for_first_img(img):
    # 使用Haar级联分类器检测眼睛
    eye_cascade = cv2.CascadeClassifier(haar_path)
    eyes = eye_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    # 如果检测到眼睛，返回眼睛的中心坐标
    if len(eyes) > 0:
        for (ex, ey, ew, eh) in eyes:
            cv2.circle(img, (ex + ew // 2, ey + eh // 2), 5, (0, 0, 255), -1)
            ptA = (ex, ey)
            ptB = (ex + ew, ey + eh)
            # cv2.imshow('image', img)
            # cv2.waitKey(1)
            print ("检测到瞳孔位置.")
        return eyes[0][0] + eyes[0][2] // 2, eyes[0][1] + eyes[0][3] // 2

    else:
        print("没有检测到位置，需要结束进程重试.")
        return None

def rotate_imgs(img,M):
    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    # 将旋转后的图像转换为灰度图
    if len(rotated_img.shape) == 2:  # 如果是灰度图像
        gray = rotated_img  # 直接使用
    elif len(rotated_img.shape) == 3:  # 如果是彩色图像
        gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    else:
        raise ValueError("输入图像的通道数不正确")
    return gray


def crop_img(img, ex, ey):
    target_width = IMG_SIZE[0] * 4  # 计算目标宽度
    target_height = IMG_SIZE[1] * 4  # 计算目标高度
    center_x, center_y = ex, ey
    start_x = center_x - target_width // 2
    end_x = start_x + target_width
    start_y = center_y - target_height // 2
    end_y = start_y + target_height

    cropped_img = img[start_y:end_y, start_x:end_x]
    return cropped_img


def blink_detect(img,blink_model):
    
    resize_img = cv2.resize(img, (IMG_SIZE[0], IMG_SIZE[1]))

    # 检查图像的通道数
    if len(resize_img.shape) == 2:  # 如果是灰度图像
        gray = resize_img  # 直接使用
    elif len(resize_img.shape) == 3:  # 如果是彩色图像
        gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    else:
        raise ValueError("输入图像的通道数不正确")

    eye_input = gray.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255
    eye_input = tf.convert_to_tensor(eye_input)
    
    # 使用模型进行预测
    pred = blink_model(eye_input, training=False)
    pred = pred.numpy()
    
    # print(f"Prediction: {pred}")
    return pred



def process_images(img_dir, blink_model, output_dir):

    # 1. 图像处理获取原始数据
    data, name = load_image(img_path=img_dir, flag=False)

    # 2. 旋转图像
    M = rotate_first_img(data[0])
    for i in range(len(data)):
        data[i] = rotate_imgs(data[i], M)

    # 3. 检测瞳孔
    ex,ey = detect_pupil_for_first_img(data[0])

    # 3.5 保存下第一张Crop图像
    dir = os.path.join(img_dir,)
    crop_first_img = crop_img(data[0], ex ,ey)
    cv2.imwrite(output_dir, crop_first_img)
    
    # 4. 裁剪图像
    cropped_data = [crop_img(data[i], ex, ey) for i in range(len(data))]

    # 5. 眨眼检测
    blink_p = []
    for i in tqdm(range(len(cropped_data))):  # 使用 tqdm 来显示进度条
        result = blink_detect(cropped_data[i],blink_model)
        blink_p.append(result)

    # 6. 保存结果
    df = pd.DataFrame({'Image Name': name, 'Prediction': blink_p})
    df.to_csv('EyeFeatureDetect/predictions.csv', index=False)
    return df, cropped_data



if __name__ == "__main__":
    img_path_single = "EyeTrackData/1/2025-02-24_16-15-56_1.png"
    img_single = cv2.imread(img_path_single)
    M = rotate_first_img(img_single)
    img_single = rotate_imgs(img_single,M)
    ex,ey = detect_pupil(img_single)
    
    img_path = "EyeTrackData/1/"
    data, name = load_image(img_path=img_path, flag=False)

    np.save('EyeFeatureDetect/rotation_matrix.npy', M)  # 保存旋转矩阵M
    np.save('EyeFeatureDetect/pupil_coordinates.npy', [ex, ey])  # 保存瞳孔坐标ex和ey

    blink_p = []
    img_names = []  # 用于保存图像名称
    for img, name in zip(data, name):  # 同时遍历图像和名称
        img = rotate_imgs(img,M)  #先旋转
        cropped_img = crop_img(img,ex,ey)
        p = blink_detect(cropped_img)
        blink_p.append(p)
        img_names.append(name)  # 保存图像名称

    # 保存为CSV文件
    df = pd.DataFrame({'Image Name': img_names, 'Prediction': blink_p})  # 创建DataFrame
    df.to_csv('EyeFeatureDetect/predictions.csv', index=False)  # 保存为CSV文件


   

