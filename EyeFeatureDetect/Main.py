import os
import pandas as pd
from blink_statistics import analyze_blinks
from CorrectionPhoto import process_images
from PupilDetection import pupil_detection
import json
from tensorflow.python.keras.models import load_model

#-----------------------------加载model------------------------------------
global blink_model 
blink_model = load_model("PostProcess/Models/2018_12_17_22_58_35.h5")
blink_model.summary()

#------------------------------Main-------------------------------------------
def main(dir_name,output_dir, times):
    # 配置参数
    config = {
        "input_dir": f"{dir_name}",
        "output_dir": f"{output_dir}",
        "times": f"{times}",
        "haar_cascade": "PostProcess/Models/Haar/haarcascade_eye.xml",
        "blink_model": "PostProcess/Models/2018_12_17_22_58_35.h5",
        "window_size": 20,
        "Pupil_radius_plot_enabled": False,
        "Pupil_radius_single_plot": False
    }
    csv_dir = os.path.join(f"{output_dir}", "CSV_"f"{times}")
    # 创建输出目录
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    try:
        # 第一阶段：图像处理获取原始数据
        print("开始图像处理...")
        raw_df, cropped_data = process_images(
            img_dir=config["input_dir"],
            blink_model = blink_model,
            output_dir = os.path.join(config["output_dir"], f"{times}_first_img.png")
        )
        raw_output = os.path.join(csv_dir, f"{times}_raw_predictions.csv")
        raw_df.to_csv(raw_output, index=False)
        print(f"原始数据已保存至：{raw_output}")

        # 第二阶段：数据分析生成报告
        print("\n开始数据分析...")
        
        analyzed_df, blink_events = analyze_blinks(raw_df)
        analyzed_output = os.path.join(csv_dir, f"{times}_analyzed_results.csv")
        blink_output = os.path.join(csv_dir, f"{times}_blink_events.csv")
        analyzed_df.to_csv(analyzed_output, index=False)
        blink_events.to_csv(blink_output, index=False)
        print(f"分析结果已保存至：{analyzed_output}")
        print(f"眨眼事件报告已保存至：{blink_output}")

        # 第三阶段：瞳孔运动分析
        print("\n开始瞳孔运动分析...")
        result_df, total_move, stats = pupil_detection(cropped_data, analyzed_df, config)
        analyzed_output = os.path.join(csv_dir, f"{times}_analyzed_results.csv")
        result_df.to_csv(analyzed_output, index=False)
        print(f"分析结果已保存至：{analyzed_output}")

        # 显示简要统计
        print("\n分析摘要：")
        print(f"处理总帧数：{len(analyzed_df)}")
        print(f"检测到眨眼事件：{len(blink_events)}次")
        if len(blink_events) > 0:
            avg_duration = blink_events['duration_ms'].str.replace('ms','').astype(float).mean()
            total_duration = blink_events['duration_ms'].str.replace('ms','').astype(float).sum()
            print(f"平均眨眼时长：{avg_duration:.1f}ms")
            print(f"总眨眼时长：{total_duration:.1f}ms")
        print(f"瞳孔运动统计：{stats}")
        print(f"总移动距离：{total_move:.2f}像素")
        print(f"瞳孔半径：{stats['mean_radius']:.2f}像素")

        # 保存统计数据为 JSON 文件
        json_output = os.path.join(config["output_dir"], f"{times}_statistics_summary.json")
        stats_data = {
            "文件名": dir_name,
            "总帧数": len(analyzed_df),
            "眨眼事件数": len(blink_events),
            "平均眨眼时长(ms)": avg_duration if len(blink_events) > 0 else 0,
            "总移动距离(像素)": total_move,
            "瞳孔半径_Mean(像素)": stats['mean_radius'],
            "总移动距离(像素)": total_move,
            "瞳孔运动统计": stats
        }
        with open(json_output, 'w', encoding='utf-8') as json_file:
            json.dump(stats_data, json_file, ensure_ascii=False, indent=4)
        print(f"统计数据已保存至：{json_output}")

    except Exception as e:
        print(f"\n处理过程中发生错误：{str(e)}")
        if 'raw_df' in locals():
            error_output = os.path.join(config["output_dir"], "error_snapshot.csv")
            raw_df.to_csv(error_output)
            print(f"错误发生时数据快照已保存至：{error_output}")
    

if __name__ == "__main__":
    dir_name = "E:/EyeTrackData/ZLP_0308_Ex2_000"
    output_dir = os.path.join(dir_name, "AnalysisResults")
    # ex1: 7 times   ex2 choose 1,4,7,10 time

    n = 1 
    sub_folder = os.path.join(dir_name, str(n))  # 拼接子文件夹路径
    output = os.path.join(dir_name,output_dir)
    main(sub_folder, output_dir=output, times = n)
    # main(dir_name=dir_name, output_dir=output_dir,times=n)
    