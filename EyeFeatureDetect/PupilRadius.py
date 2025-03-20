import cv2
import numpy as np
import CorrectionPhoto as cp

def blob_detection(gray, lum=25, blob_minsize=10, blob_maxsize=100, min_circularity=0.5):
    
   # 自适应阈值化
    _, binary_inv = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    
    # 形态学操作去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # 计算当前轮廓的面积
        area = cv2.contourArea(cnt)
        
        # 检查面积是否在指定的最小和最大范围内
        if area < blob_minsize**2 * np.pi / 4 or area > blob_maxsize**2 * np.pi / 4:
            continue  # 如果不在范围内，跳过当前轮廓
        
        # 计算当前轮廓的周长
        perimeter = cv2.arcLength(cnt, True)
        
        # 如果周长为0，跳过当前轮廓
        if perimeter == 0:
            continue
        
        # 计算轮廓的圆度
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # 检查圆度是否大于0.7，表示该轮廓接近圆形
        if circularity > min_circularity:
            # 计算当前轮廓的最小外接圆的中心和半径
            (x, y), radius = cv2.minEnclosingCircle(cnt)
        
    if radius is not None:
        return radius  # 返回找到的半径
    else:
        return None


if __name__ == "__main__":
    img_path = "EyeTrackData/1/2025-02-24_16-16-45_4191.png"
    image_gray = cv2.imread(img_path)
    image = image_gray.copy()
    
    M = cp.rotate_first_img(image_gray)
    image = cp.rotate_imgs(image, M)
    image_gray = cp.rotate_imgs(image_gray, M)
    ex,ey = cp.detect_pupil(image_gray)

    cropped_data = cp.crop_img(image, ex, ey) 
    
    radius = blob_detection(cropped_data, 127)
    print(radius)
