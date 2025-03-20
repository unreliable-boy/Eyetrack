import cv2
# import dlib
import numpy as np
from imutils import face_utils
from tensorflow.python.keras.models import load_model
import tensorflow as tf
import os

IMG_SIZE = (34, 26)

model = load_model('D:/EyeTrack_Break/PostProcess/Models/2018_12_17_22_58_35.h5')
model.summary()

# 打开输入视频
cap = cv2.VideoCapture('VideoTest/Eye_crop.mp4')
cap_all = cv2.VideoCapture('VideoTest/eye_tracking.mp4')

# 获取视频属性
frame_width = int(cap_all.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap_all.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap_all.get(cv2.CAP_PROP_FPS))

# 创建视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或使用 'XVID'
out = cv2.VideoWriter('output_with_blink.mp4', fourcc, fps, (frame_width, frame_height))
data = []
while cap.isOpened() and cap_all.isOpened():
    ret, img_ori = cap.read()
    ret_all, img_all = cap_all.read()
    if not ret or not ret_all:
        break
    
    img = img_ori.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eye_img = gray.copy()
    eye_img = cv2.resize(eye_img, (IMG_SIZE[0], IMG_SIZE[1]))
    print(f"eye_img shape: {eye_img.shape}")
    
    # 将输入转换为tensorflow张量
    eye_input = eye_img.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255
    eye_input = tf.convert_to_tensor(eye_input)
    
    # 使用模型进行预测
    pred = model(eye_input, training=False)
    pred = pred.numpy()
    
    print(f"Prediction: {pred}")
    pred_text = f"Blink: {pred[0][0]:.2f}"
    cv2.putText(img_all, pred_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 写入帧到输出视频
    data.append(img_all)
    
    # 可选：预览视频
    cv2.imshow('Preview', img_all)
    if cv2.waitKey(1) == ord('q'):
        break

# 释放资源
cap.release()
cap_all.release()
cv2.destroyAllWindows()

output_dir = "VideoTest"
height, width = data[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式编码
out_path = os.path.join(output_dir, "blink_detection.mp4")
out = cv2.VideoWriter(out_path, fourcc, 60.0, (width, height))
    
    # 将图像序列写入视频
for frame in data:
    out.write(frame)
    
    # 释放视频写入器
out.release()
   
    

    
    