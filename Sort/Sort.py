import os
import shutil
from datetime import datetime, timedelta

# 假设你已经有一个包含图片文件的目录

# output_directory = "Z:/Master_Theses/EyeTrackData/WWB_0223_Ex1_015/Save"
#image_directory = "Sort/ZZY_0215_Ex2_000/Right"
# output_directory = "Sort/ZZY_0215_Ex2_000/Save"

def sort_images_by_timestamp(image_directory):
    # 获取所有图片文件
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.png')]

    # 根据文件名中的时间戳提取时间
    def extract_timestamp(filename):
        timestamp_str = filename.split('_')[0] + '_' + filename.split('_')[1]  # 获取时间戳部分
        return datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")

    output_directory = os.path.join(os.path.dirname(image_directory), "Save")  # 输出目录设置为输入目录同级的Save文件夹
    # 排序文件列表，根据时间戳排序
    image_files.sort(key=lambda f: extract_timestamp(f))

    # 分组逻辑：使用一个阈值来决定何时分组
    time_threshold = timedelta(minutes=1)  # 设置1分钟为一个分组的最大时间间隔
    current_group = []
    last_timestamp = None

    # 遍历所有图片文件，按时间分组
    for image_file in image_files:
        timestamp = extract_timestamp(image_file)
        
        # 如果这是第一张图片，或者与上张图片的时间间隔小于阈值，继续分在同一组
        if last_timestamp is None or timestamp - last_timestamp <= time_threshold:
            current_group.append(image_file)
        else:
            # 如果时间间隔超过阈值，创建新的文件夹并移动当前组的文件
            group_folder = os.path.join(output_directory, f"group_{current_group[0].split('_')[0]}_{current_group[0].split('_')[1]}")
            os.makedirs(group_folder, exist_ok=True)
            
            for file in current_group:
                shutil.move(os.path.join(image_directory, file), os.path.join(group_folder, file))
            
            # 开始新的分组
            current_group = [image_file]

        # 更新上一次的时间戳
        last_timestamp = timestamp

    # 最后一组文件移动
    if current_group:
        group_folder = os.path.join(output_directory, f"group_{current_group[0].split('_')[0]}_{current_group[0].split('_')[1]}")
        os.makedirs(group_folder, exist_ok=True)
        
        for file in current_group:
            shutil.move(os.path.join(image_directory, file), os.path.join(group_folder, file))

    print("分类完成!")

if __name__ == "__main__":
    dir1 = "E:/EyeTrackData/DUDU_0226_Ex1_100/Right"
    dir2 = "E:/EyeTrackData/DUDU_0226_Ex1_065/Right"
    # dir3 = "E:/EyeTrackData/ZLP_0308_Ex1_065/Right"
    # dir4 = "E:/EyeTrackData/ZLP_0308_Ex1_015/Right"

    sort_images_by_timestamp(dir1)
    print("1")
    sort_images_by_timestamp(dir2)
    print("2")
    #sort_images_by_timestamp(dir3)
    #print("3")
   # sort_images_by_timestamp(dir4)
 


