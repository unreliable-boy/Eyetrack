import numpy as np
import os
import cv2

def BLINK(self, max_len=300):
    """
    眨眼检测函数
    通过分析图像亮度变化来检测眨眼动作
    参数:
        max_len: 用于计算的最大帧数，默认300帧
    返回:
        blinkvalue: 眨眼状态值（0.0表示睁眼，0.8表示闭眼）
    """
    # 如果需要清除眨眼检测状态
    if self.blink_clear == True:
        self.max_ints = []        # 清空最大亮度列表
        self.max_int = 0          # 重置最大亮度值
        self.frames = 0           # 重置帧计数器

    # 计算当前帧的总亮度
    intensity = np.sum(self.current_image_gray_clean)

    # 管理滑动窗口的亮度值列表
    if self.calibration_frame_counter == max_len:
        self.filterlist = []      # 达到最大帧数时清空过滤器
    if len(self.filterlist) < max_len:
        self.filterlist.append(intensity)    # 添加新的亮度值
    else:
        self.filterlist.pop(0)              # 移除最旧的值
        self.filterlist.append(intensity)    # 添加新的值

    # 异常值处理：如果当前亮度值异常（太高或太低）
    if (
        intensity >= np.percentile(self.filterlist, 99)
        or intensity <= np.percentile(self.filterlist, 1)
        and len(self.max_ints) >= 1
    ):
        try:
            intensity = min(self.max_ints)   # 使用历史最小值替代异常值
        except:
            pass

    # 更新帧计数和最大/最小亮度值
    self.frames = self.frames + 1
    if intensity > self.max_int:
        self.max_int = intensity
        if self.frames > max_len:    # 如果帧数超过阈值
            self.max_ints.append(self.max_int)   # 记录最大值
    if intensity < self.min_int:
        self.min_int = intensity     # 更新最小值

    # 眨眼状态判断
    if len(self.max_ints) > 1:
        if intensity > min(self.max_ints):
            blinkvalue = 0.0    # 睁眼状态
        else:
            blinkvalue = 0.8    # 闭眼状态
    try:
        return blinkvalue
    except:
        return 0.8    # 如果出错，默认返回闭眼状态


if __name__ == "__main__":
    # 使用相对路径或环境变量
    imt_path = os.path.join("EyeTrackData", "1", "Right", "Right")
    
    
    

