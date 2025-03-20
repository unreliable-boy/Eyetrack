import dlib
import cv2
import numpy as np
import os


def shape_to_np(shape, dtype="int"): # 将包含68个特征的的shape转换为numpy array格式
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def detector(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("D:/EyeTrack/PostProcess/Models/shape_predictor_68_face_landmarks.dat")
    image = cv2.resize(image, (400, 400))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    shapes = []
    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        shapes.append(shape)
    print(rects)
    for shape in shapes:
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
    
    cv2.imshow("Output", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    image = cv2.imread("EyeTrackData/3/Right/2025-01-05_13-00-46_1505.png")
    image = cv2.imread("D:/EyeTrack/PostProcess/Models/open_man.jpg")
    detector(image)



