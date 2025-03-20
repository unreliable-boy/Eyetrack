import CorrectionPhoto as cp
import cv2
import numpy as np
import re
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from PupilRadius import blob_detection

def calculate_pupil_movement(df, pupil_array):
    """
    将瞳孔运动数据整合到DataFrame并计算移动距离
    
    参数：
    result_df: 原始数据框（必须包含'label'列）
    pupil_array: 瞳孔位置数组，形状为(N,2)
    
    返回：
    result_df: 新增两列（pupil_position, movement_delta）的数据框
    """
    # 创建副本避免修改原始数据
    pupil_array = np.array(pupil_array)  
    # 添加瞳孔位置列（字符串格式）
    df['pupil_position'] = None  # 初始化为None
    # df['pupil_position'] = df['pupil_position'].astype('object')
    df['movement_delta'] = None  # 浮点数列
    # 提取label为'F'的行并记录行数
    valid_mask = (df['label'] == 'F').values
    valid_indices = np.where(valid_mask)[0]


    # 检查数组维度匹配
    if len(valid_indices) != pupil_array.shape[0]:
        raise ValueError("瞳孔数组与有效帧数量不匹配，"
                        f"预期{len(valid_indices)}个，实际收到{pupil_array.shape[0]}个")

    # ================== 填充瞳孔位置 ==================
    for arr_idx, df_idx in enumerate(valid_indices):
        x, y = pupil_array[arr_idx]
        # 处理无效坐标（0,0）或空行
        if x == 0 and y == 0:
            df.at[df_idx, 'pupil_position'] = "INVALID"
        else:
            df.at[df_idx, 'pupil_position'] = f"{x:.2f},{y:.2f}"

    # 填充其他行为空行为"INVALID"
    df['pupil_position'] = df['pupil_position'].fillna("INVALID")

    

    # ================== 计算移动距离 ==================
    prev_valid_pos = None  # 存储前一个有效坐标 (x,y)
    total_movement = 0.0

    # 仅遍历有效帧（提升性能）
    for df_idx in valid_indices:
        current_str = df.at[df_idx, 'pupil_position']

        if current_str == "INVALID":
            continue
        try:
            # 从字符串解析坐标
            current_x, current_y = map(float, current_str.split(','))
        except:
            continue

        # 只有当存在前一个有效坐标时才计算距离
        if prev_valid_pos is not None:
            dx = current_x - prev_valid_pos[0]
            dy = current_y - prev_valid_pos[1]
            distance = np.hypot(dx, dy)
            # 更新当前帧的移动距离
            df.at[df_idx, 'movement_delta'] = distance
            total_movement += distance
            # 更新前一个有效坐标（无论是否计算过距离）
        prev_valid_pos = (current_x, current_y)

        # ================== 数据后处理 ==================
    # 统一替换无效标记为NaN
    df['pupil_position'] = df['pupil_position'].replace("INVALID", np.nan)
    
    # 强制类型转换确保数据一致性
    df['movement_delta'] = pd.to_numeric(df['movement_delta'], errors='coerce')
    df['pupil_radius'] = pd.to_numeric(df['pupil_radius'], errors='coerce')
    
    return df, round(total_movement, 4)
        

def pupil_detection(data, input_df, config):
    # 调试信息：检查 data 的长度
    print("data 的长度:", len(data))
    cascade_path = config["haar_cascade"]
    window_size = config["window_size"]
    plot_enabled = config["Pupil_radius_plot_enabled"]
    single_plot = config["Pupil_radius_single_plot"]
    times = config["times"]
    picture_output_dir = os.path.join(config["output_dir"], f"{times}_PupilRadius.png")

    df = input_df.copy()
    df.insert(df.shape[1], 'pupil_position', None)
    df.insert(df.shape[1], 'movement_delta', None)
    df.insert(df.shape[1], 'pupil_radius', None)

    # 提取有效索引
    valid_mask = (input_df['label'] == 'F').values
    valid_indices = np.where(valid_mask)[0]

    # 获取有效图像
    valid_images = [data[i] for i in valid_indices]  # 这里可能会引发 IndexError

    # 初始化存储
    pupil_positions = []
    failed_frames = 0
    pupil_radius = []

    # 配置瞳孔检测参数
    eye_cascade = cv2.CascadeClassifier(cascade_path)
    detect_params = {
        'scaleFactor': 1.1,
        'minNeighbors': 5,
        'minSize': (30, 30)
    }
    # 遍历有效帧
    for idx, img in tqdm(enumerate(valid_images), total=len(valid_images), desc="Processing Images"):
        # 转换为灰度图（如果尚未转换）
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # 瞳孔检测
        eyes = eye_cascade.detectMultiScale(gray, **detect_params)
        if len(eyes) > 0:
            x, y, w, h = eyes[0]
            center = (x + w//2, y + h//2)
            pupil_positions.append(center)
        else:
            failed_frames += 1
            # 使用线性插值填充缺失值（可选）
            if len(pupil_positions) > 0:
                pupil_positions.append(pupil_positions[-1])
            else:
                pupil_positions.append((0, 0))  # 默认值
            # 转换为numpy数组
        
 
    pupil_array = np.array(pupil_positions)
    for idx, img in tqdm(enumerate(valid_images), total=len(valid_images), desc="Pupil Radius Detection"):
    # 使用blob_detection函数检测瞳孔直径
        radius = blob_detection(gray)
        if radius is not None:
            pupil_radius.append((radius))
        else:
            pupil_radius.append((None))
    # ================== 瞳孔半径数据处理 ==================
    # 计算瞳孔半径的平均值以及填入表格
    total_radius = 0
    radius_count = 0
    
    pupil_radius = np.array(pupil_radius)
    
    # 数据长度验证
    if len(pupil_radius) != len(valid_indices):
        raise ValueError(f"瞳孔半径数据长度({len(pupil_radius)})与有效帧数({len(valid_indices)})不匹配")
    try:
    # 将数据写入 DataFrame
        for arr_idx, df_idx in enumerate(valid_indices):
            df.at[df_idx, 'pupil_radius'] = pupil_radius[arr_idx]
            if pupil_radius[arr_idx] is not None:
                total_radius += pupil_radius[arr_idx]
                radius_count += 1
        mean_radius = total_radius / radius_count
        df['pupil_radius'] = df['pupil_radius'].fillna("INVALID")
        pupil_radius_filtered = mean_radius_filter(pupil_radius, window_size)
        # 写入数据
        for arr_idx, df_idx in enumerate(valid_indices):
            df.at[df_idx, 'pupil_radius_filtered'] = pupil_radius_filtered[arr_idx]

        # 绘制图像
        plot_pupil_radius(pupil_radius, pupil_radius_filtered, single_plot, picture_output_dir)
                
    except Exception as e:
        print("瞳孔半径数据处理失败",e)


    # 计算统计指标
    stats = {
        'total_frames': len(valid_images),
        'success_rate': (len(valid_images) - failed_frames) / len(valid_images),
        'mean_position': None,
        'std_deviation': None,
        'movement_range': None,
        'total_movement': None,
        'mean_radius': None
    }

   #=======================统计瞳孔运动长度=========================== 
    result_df, total_move = calculate_pupil_movement(df, pupil_array)
    if len(pupil_array) > 0:
        stats['mean_position'] = np.mean(pupil_array, axis=0).tolist()
        stats['std_deviation'] = np.std(pupil_array, axis=0).tolist()
        stats['movement_range'] = {
            'x': (pupil_array[:,0].min(), pupil_array[:,0].max()),
            'y': (pupil_array[:,1].min(), pupil_array[:,1].max())
        }
        stats['total_movement'] = total_move
    if radius_count > 0:
        stats['mean_radius'] = mean_radius

    return result_df, total_move, stats

# ================== 瞳孔半径数据处理(中值滤波) ==================

def mean_radius_filter(pupil_radius, window_size):
    # 将None值替换为NaN以便进行数值处理
    pupil_radius_np = np.array(pupil_radius, dtype=float)
    # 找出有效的半径值
    valid_mask = ~np.isnan(pupil_radius_np)
    valid_radius = pupil_radius_np[valid_mask]
    if len(valid_radius) > 0:
        # 应用中值滤波 (只对有效值进行滤波)
        filtered_radius = medfilt(valid_radius, kernel_size=window_size)
        # 将滤波后的值放回原数组
        pupil_radius_filtered = pupil_radius_np.copy()
        pupil_radius_filtered[valid_mask] = filtered_radius
    return pupil_radius_filtered

# ================== 瞳孔半径数据处理(绘制图像) ==================
def plot_pupil_radius(pupil_radius_np,pupil_radius_filtered,single_plot,output_dir):
    valid_mask = ~np.isnan(pupil_radius_np)
    valid_radius = pupil_radius_np[valid_mask]
    pupil_mean = np.mean(valid_radius)
    # 设置y轴刻度
    y_ticks = [
        np.min(valid_radius),
        (np.min(valid_radius) + pupil_mean) / 2,
        pupil_mean,
        (np.max(valid_radius) + pupil_mean) / 2,
        np.max(valid_radius)
    ]
    if single_plot:
        # 在一个图中绘制
        plt.figure(figsize=(12, 6))
        plt.plot(pupil_radius_np, alpha=0.5, label='原始数据', color='lightblue')
        plt.plot(pupil_radius_filtered, label='滤波后数据', color='darkblue')
        plt.yticks(y_ticks)
        plt.xlabel('帧')
        plt.ylabel('瞳孔大小')
        plt.title('瞳孔大小数据 - 原始 vs 滤波后')
        plt.legend()
        plt.grid(True)
        # 保存图像
        plt.savefig('PupilRadius.png')
    else:
        # 在两个子图中分别绘制
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(pupil_radius_np, color='lightblue')
        ax1.set_yticks(y_ticks)
        ax1.set_xlabel('帧')
        ax1.set_ylabel('瞳孔大小')
        ax1.set_title('原始瞳孔大小数据')
        ax1.grid(True)
        
        ax2.plot(pupil_radius_filtered, color='darkblue')
        ax2.set_yticks(y_ticks)
        ax2.set_xlabel('帧')
        ax2.set_ylabel('瞳孔大小')
        ax2.set_title('滤波后瞳孔大小数据')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    input_df = pd.read_csv("AnalysisResults/analyzed_results.csv")
    img_path = "EyeTrackData/1"
    data, name = cp.load_image(img_path)
    result_df, total_move, stats = pupil_detection(data, input_df, "PostProcess/Models/Haar/haarcascade_eye.xml")


