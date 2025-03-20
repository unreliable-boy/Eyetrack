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

def get_correction_matrix(img, corners):
    # 确保corners是正确的格式（4个点，float32类型）
    pts_o = np.float32(corners)  # 转换为float32类型
    
    board_size = (3000, 3000)
    corner_size = 500
    center_x = board_size[0] // 2
    center_y = board_size[1] // 2
    pts_d = np.float32([
        [center_x - corner_size//2, center_y - corner_size//2],  # 左上
        [center_x + corner_size//2, center_y - corner_size//2],  # 右上
        [center_x + corner_size//2, center_y + corner_size//2],  # 右下
        [center_x - corner_size//2, center_y + corner_size//2]   # 左下
    ])
    
    # 确保两个点数组都是4x2的形状
    pts_o = pts_o.reshape((4, 2))
    pts_d = pts_d.reshape((4, 2))
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts_o, pts_d)
    
    # 创建一个大画布
    board = np.zeros((board_size[0], board_size[1], 3), dtype=np.uint8)
    
    # 在大画布上进行透视变换
    dst = cv2.warpPerspective(img, M, board_size)
    
    # 计算中心位置
    y_offset = (board_size[0] - dst.shape[0]) // 2
    x_offset = (board_size[1] - dst.shape[1]) // 2
    
    # 将变换后的图像放在画布中心
    board[y_offset:y_offset+dst.shape[0], x_offset:x_offset+dst.shape[1]] = dst
    board = cv2.resize(board, (1000, 1000))
    return board


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
    cv2.imshow("img_corrected", img_corrected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_corrected
    

def main():
    img_L_Path = "EyeTrackData/Calibration/Right/2025-01-07_16-50-40_526.png"
    img_L = cv2.imread(img_L_Path)
    img_L = cv2.resize(img_L, (1000, 1000))
    corners = Find_corners(img_L)
    H_L = get_correction_matrix(corners)
    
    return H_L


  

if __name__ == "__main__":
    # 读取图像
    img_path = "EyeTrackData/Calibration/Left/2025-01-06_17-16-09_450.png"
    img_L_Path = "EyeTrackData/Calibration/Left/2025-01-07_16-50-40_526.png"
    img_R_Path = "EyeTrackData/Calibration/Right/2025-01-07_16-50-40_526.png"
    
    img_L = cv2.imread(img_L_Path)
    img_R = cv2.imread(img_R_Path)
    img = cv2.imread(img_path)
    img_R = cv2.rotate(img_R, cv2.ROTATE_90_CLOCKWISE)
    
    # 调整图像大小
    # img = cv2.resize(img, (1000, 1000))
    img_L = cv2.resize(img_L, (1000, 1000))
    img_R = cv2.resize(img_R, (1000, 1000))
    
    corner_L = np.array([[438, 392], [511, 180], [653, 364], [511, 549]])
    
    img_corrected_L =get_correction_matrix(img_L,corner_L)
    cv2.imshow("img_corrected_L", img_corrected_L)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    
    
    
    test_img_path = "EyeTrackData/3/Left/2025-01-05_13-00-46_1505.png"
    test_img = cv2.imread(test_img_path)
    if test_img is not None:
        test_img = cv2.resize(test_img, (1000, 1000))
        # corrected_test = warpimg(test_img, H_L)
        img_corrected = get_correction_matrix(test_img,corner_L)
        cv2.imshow("img_corrected", img_corrected)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    

    
    
    