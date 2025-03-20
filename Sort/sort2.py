import os
import shutil
from datetime import datetime as dt

def parse_time_from_filename(filename):
    # 分割前两个下划线以提取日期和时间部分
    parts = filename.split('_', 2)
    if len(parts) < 3:
        raise ValueError(f"Invalid filename format: {filename}")
    date_part, time_part, _ = parts
    time_str = f"{date_part}_{time_part}"
    return dt.strptime(time_str, "%Y-%m-%d_%H-%M-%S")

def organize_images(src_dir, dst_parent=None):
    # 设置目标目录（默认与源目录同级）
    if dst_parent is None:
        dst_parent = os.path.join(src_dir, "organized")
    os.makedirs(dst_parent, exist_ok=True)

    # 获取所有PNG文件
    files = [f for f in os.listdir(src_dir) if f.lower().endswith('.png')]
    if not files:
        print("No PNG files found in the source directory.")
        return

    # 解析时间并查找最小时间t0
    file_times = []
    t0 = None
    for f in files:
        try:
            t = parse_time_from_filename(f)
            file_times.append((f, t))
            if t0 is None or t < t0:
                t0 = t
        except ValueError as e:
            print(f"Skipping invalid file {f}: {e}")

    if not file_times:
        print("No valid files with parsable timestamps.")
        return

    # 移动文件到对应文件夹
    for filename, file_time in file_times:
        delta_seconds = (file_time - t0).total_seconds()
        folder_num = int(delta_seconds // 60) + 1  # 从1开始编号
        
        target_dir = os.path.join(dst_parent, str(folder_num))
        os.makedirs(target_dir, exist_ok=True)
        
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(target_dir, filename)
        shutil.move(src_path, dst_path)
        print(f"Moved {filename} -> {target_dir}")

if __name__ == "__main__":
    # source_directory = "Z:/Master_Theses/EyeTrackData/ZZY_0215_Ex2_000/Right"
    dir1 = "E:/EyeTrackData/ZLP_0308_Ex2_100/Right" 
    dir2 = "E:/EyeTrackData/ZLP_0308_Ex2_065/Right" 
    dir3 = "E:/EyeTrackData/ZLP_0308_Ex2_015/Right"
    dir4 = "E:/EyeTrackData/ZLP_0308_Ex2_000/Right"
    # organize_images(source_directory)
    organize_images(dir1)
    organize_images(dir2)
    organize_images(dir3)
    organize_images(dir4)
