import cv2
import numpy as np

def Find_corners(img): 
    # 检查输入图像是否为空
    if img is None:
        return None

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)
    
    # 设置棋盘格模式
    checkboard_size = (5,5)
    
   

    # 组合多个标志位以提高检测准确性
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH +  # 使用自适应阈值
        cv2.CALIB_CB_NORMALIZE_IMAGE +  # 归一化图像亮度
        cv2.CALIB_CB_FAST_CHECK        # 快速检查模式
    )
    
    # 尝试不同的阈值进行检测
    ret, corners = cv2.findChessboardCornersSB(
        gray,
        checkboard_size,
        flags=cv2.CALIB_CB_EXHAUSTIVE
    )
    # 如果找到棋盘格角点
    if ret:
        # 使用更精确的参数进行角点精确化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        corners = cv2.cornerSubPix(
            gray, 
            corners, 
            (11,11),  # 搜索窗口大小
            (-1,-1),  # 死区大小
            criteria
        )
        
        # 绘制检测到的角点
        img_with_corners = img.copy()
        # cv2.drawChessboardCorners(img_with_corners, checkboard_size, corners, ret)
        rows, cols = checkboard_size
        # 将corners重塑为rows x cols的数组
        corners_reshaped = corners.reshape(rows, cols, 2)
        # 转置数组，交换行和列
        corners_transposed = corners_reshaped.transpose(1, 0, 2)
        # 将角点数据重塑为一维向量
        corners = corners_transposed.reshape(-1, 2)
        img_with_corners = cv2.drawChessboardCorners(img_with_corners, checkboard_size, corners, ret)
        cv2.imshow("Output", img_with_corners)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return corners

def get_correction_matrix(corners):
    if corners is None:
        return None
    checkboard_size = (5,5)
    # 计算理想棋盘格角点的位置
    square_size = 150  # 每个方格的像素大小
    
    board_size = (3000, 3000)  
    pattern_width = checkboard_size[0] * square_size
    pattern_height = checkboard_size[1] * square_size
    
    # 计算棋盘格在2000x2000画布上的中心位置
    start_x = (board_size[0] - pattern_width) // 2
    start_y = (board_size[1] - pattern_height) // 2
    
    # 生成居中的pattern_points
    pattern_points = np.zeros((checkboard_size[0] * checkboard_size[1], 2), np.float32)
    for i in range(checkboard_size[1]):
        for j in range(checkboard_size[0]):
            pattern_points[i * checkboard_size[0] + j] = [
                start_x + j * square_size,  # x坐标
                start_y + i * square_size   # y坐标
            ]
    # 计算单应性矩阵
    H, _ = cv2.findHomography(corners, pattern_points)
    return H 
    


def warpimg(img, H):
    img = cv2.resize(img, (1000, 1000))
    board_size = (3000, 3000)
    board = np.zeros((board_size[0], board_size[1], 3), dtype=np.uint8)
        
        # 计算原始图像变换后的大小
    warped_img = cv2.warpPerspective(img, H, board_size)
            # 计算中心位置，使变换后的图像位于画布中央
    y_offset = (board_size[0] - warped_img.shape[0]) // 2
    x_offset = (board_size[1] - warped_img.shape[1]) // 2
        
        # 将变换后的图像放置到画布中央
    board[y_offset:y_offset+warped_img.shape[0], 
          x_offset:x_offset+warped_img.shape[1]] = warped_img
        
        # 更新图像为board
    img_corrected = board
    img_corrected = cv2.resize(img_corrected, (1000, 1000))
    return img_corrected
    

def main():
    img_L_Path = "EyeTrackData/Test/Left/2025-01-08_16-58-32_115.png"
    img_L = cv2.imread(img_L_Path)
    img_L = cv2.resize(img_L, (1000, 1000))
    corners = Find_corners(img_L)
    H_L = get_correction_matrix(corners)
    
    return H_L


  

if __name__ == "__main__":
    # 读取图像
    img_path = "EyeTrackData/Calibration/Left/2025-01-06_17-16-09_450.png"
    # img_L_Path = "EyeTrackData/Calibration/Left/2025-01-07_16-50-40_526.png"
    img_R_Path = "EyeTrackData/Calibration/Right/2025-01-07_16-50-40_526.png"
    img_L_Path = "EyeTrackData/Test/Left/2025-01-08_16-58-32_115.png"
    
    
    img_L = cv2.imread(img_L_Path)
    img_R = cv2.imread(img_R_Path)
    img = cv2.imread(img_path)
    img_R = cv2.rotate(img_R, cv2.ROTATE_90_CLOCKWISE)
    
    # 调整图像大小
    # img = cv2.resize(img, (1000, 1000))
    img_L = cv2.resize(img_L, (1000, 1000))
    img_R = cv2.resize(img_R, (1000, 1000))
    
    
    # 获取校准矩阵
    # img_corrected, H = ImageCorrection(img)
    corners_L = Find_corners(img_L)
    # corners_R = Find_corners(img_R)
    print(corners_L)
    
    H_L = get_correction_matrix(corners_L)
    img_corrected_L = warpimg(img_L, H_L)
    
    """
    # Reshape corners_L into 5x5x2 matrix
    corners_L = corners_L.reshape(5, 5, 2)
    # Flip horizontally
    corners_L = np.flip(corners_L, axis=1)
    # Reshape back to 25x2 vector
    corners_L = corners_L.reshape(-1, 2)
    H_L = get_correction_matrix(corners_L)
    H_R = get_correction_matrix(corners_R)
    
    img_corrected_L = warpimg(img_L, H_L)
    img_corrected_R = warpimg(img_R, H_R)
    
    #对画面中心裁剪出400x400的图像
    img_corrected_L = img_corrected_L[200:600, 200:600]
    img_corrected_R = img_corrected_R[200:600, 200:600]
    
    
    combined = np.hstack((img_corrected_L, img_corrected_R))
  

    """
    
    
    test_img_path = "EyeTrackData/3/Left/2025-01-05_13-00-46_1505.png"
    test_img = cv2.imread(test_img_path)
    if test_img is not None:
        test_img = cv2.resize(test_img, (1000, 1000))
        corrected_test = warpimg(test_img, H_L)
        corrected_test = cv2.rotate(corrected_test, cv2.ROTATE_90_CLOCKWISE)
        # 对画面中心裁剪出400x400的图像
        corrected_test = corrected_test[300:700, 200:600]
        cv2.imshow("Test Image Corrected", corrected_test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  
    
    

    
    
    