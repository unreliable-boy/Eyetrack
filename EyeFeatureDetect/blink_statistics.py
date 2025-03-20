import pandas as pd
import numpy as np
import ast  # 用于安全解析字符串
from datetime import datetime


def analyze_blinks(df, threshold=0.5, fixed_fps=50, frame_threshold=5):
    # 解析帧号
    def extract_frame(img):
        return int(img.split('_')[-1].split('.')[0]) 
    
    # 预处理
    df['frame'] = df['Image Name'].apply(extract_frame)
    df = df.sort_values('frame').reset_index(drop=True)
    
    def clean_p(value):
        try:
            # 提取第一个元素（适配单元素或多元素列表）
            return float(ast.literal_eval(str(value))[0][0])
        except:
            return None  # 处理异常值
    
    df['p'] = df['Prediction'].apply(clean_p)

    # 初始化标签
    df['label'] = 'F'
    df.loc[df['p'] < threshold, 'label'] = 'T'
    

    # 第一阶段：合并邻近候选帧
    groups = []
    current_group = []
    t_indices = df[df['label'] == 'T'].index.tolist()

    for idx in t_indices:
        if not current_group:
            current_group.append(idx)
            continue
            
        last_frame = df.at[current_group[-1], 'frame']
        current_frame = df.at[idx, 'frame']
        
        if (current_frame - last_frame) <= frame_threshold:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
    
    if current_group:
        groups.append(current_group)
    
    # 第二阶段：扩展处理和时间窗标记
    for group in groups:
        if not group:
            continue
            
        start = df.at[group[0], 'frame']
        end = df.at[group[-1], 'frame']
        
        # 标记时间窗
        mask = (df['frame'] >= start - frame_threshold) & (df['frame'] <= end + frame_threshold)
        window_indices = df[mask].index
        
        if len(window_indices) == 0:
            continue
        
        # 保留第一个T标记
        first_t = window_indices[0]
        df.loc[first_t, 'label'] = 'T'
        
        # 标记其他帧为O
        other_indices = window_indices.difference([first_t])
        df.loc[other_indices, 'label'] = 'O'
    
    # 解析时间戳和帧号
    # 解析基础时间（时-分-秒）
    def parse_base_time(img):
        time_part = '_'.join(img.split('_')[1:-1])  # 提取如"16-15-56"
        return datetime.strptime(time_part, "%H-%M-%S")

    # 添加时间列
    df['base_time'] = df['Image Name'].apply(parse_base_time)
    
    # 计算每帧的绝对时间（基于50fps）
    df['frame_num'] = df['Image Name'].apply(
        lambda x: int(x.split('_')[-1].split('.')[0])
    )
    
    # 计算毫秒时间戳，默认fps为50
    default_fps = 50
    df['millisecond'] = (df['base_time'].astype(np.int64) // 10**6) * 1000 + (df['frame_num'] / default_fps) * 1000  # 使用默认fps计算

    # 检查毫秒时间戳的有效性
    df['millisecond'] = df['millisecond'].clip(lower=0)  # 确保毫秒数不为负

    # ====================== 生成持续时间报告 ======================
    blink_events = []
    current_event = []
    
    for idx, row in df.iterrows():
        if row['label'] in ['T', 'O']:
            current_event.append(row)
        elif current_event:
            # 计算持续时间（基于帧数差）
            frame_diff = current_event[-1]['frame_num'] - current_event[0]['frame_num']
            duration_ms = frame_diff * (1000 / fixed_fps)  # 固定帧率计算
            
            blink_events.append({
                "start_frame": current_event[0]['frame_num'],
                "end_frame": current_event[-1]['frame_num'],
                "duration_ms": f"{duration_ms:.1f}ms",
                "start_time": datetime.fromtimestamp(
                    max(current_event[0]['millisecond']/1000, 0)  # 确保时间戳不为负
                ).strftime('%H:%M:%S.%f')[:-3],
                "end_time": datetime.fromtimestamp(
                    max(current_event[-1]['millisecond']/1000, 0)  # 确保时间戳不为负
                ).strftime('%H:%M:%S.%f')[:-3]
            })
            current_event = []
    
    # 处理最后一个未结束的事件
    if current_event:
        frame_diff = current_event[-1]['frame_num'] - current_event[0]['frame_num']
        duration_ms = frame_diff * (1000 / fixed_fps)
        blink_events.append({
            "start_frame": current_event[0]['frame_num'],
            "end_frame": current_event[-1]['frame_num'],
            "duration_ms": f"{duration_ms:.1f}ms",
            "start_time": datetime.fromtimestamp(
                max(current_event[0]['millisecond']/1000, 0)  # 确保时间戳不为负
            ).strftime('%H:%M:%S.%f')[:-3],
            "end_time": datetime.fromtimestamp(
                max(current_event[-1]['millisecond']/1000, 0)  # 确保时间戳不为负
            ).strftime('%H:%M:%S.%f')[:-3]
        })

    # 返回结果（移除临时列）
    return (
        df.drop(['base_time', 'frame_num', 'millisecond'], axis=1),
        pd.DataFrame(blink_events)
    )


if __name__ == "__main__":
    input_df = pd.read_csv("EyeFeatureDetect/predictions.csv")
    
    # 处理数据（现在返回两个结果）
    result_df, duration_report = analyze_blinks(input_df)

    # 调试：检查数据清洗结果
    print("标注结果样例:")
    print(result_df.head())
    
    print("\n眨眼持续时间报告:")
    print(duration_report)
    
    # 保存结果
    result_df.to_csv("EyeFeatureDetect/final_predictions.csv", index=False)
    duration_report.to_csv("EyeFeatureDetect/blink_durations.csv", index=False)




"""
第二步   获取出眼球的中心位置
排除眨眼的图片，
用opencv获取眼球位置
用中心位置，计算出眼球的偏移量
计算整个过程的偏移量

"""