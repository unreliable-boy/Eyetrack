import os
import onnxruntime
# import onnx
import numpy as np
import cv2
import time
import math
from queue import Queue
import threading
from one_euro_filter import OneEuroFilter
# import psutil
# from utils.misc_utils import resource_path
from pathlib import Path


"""
眨眼检测
"""

os.environ["OMP_NUM_THREADS"] = "1"

frames = 0
models = Path("Models")


def run_model(input_queue, output_queue, session):
    while True:
        frame = input_queue.get()
        if frame is None:
            break

        img_np = np.array(frame, dtype=np.float32) / 255.0
        gray_img = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]

        gray_img = np.expand_dims(np.expand_dims(gray_img, axis=0), axis=0)

        ort_inputs = {session.get_inputs()[0].name: gray_img}
        pre_landmark = session.run(None, ort_inputs)
        pre_landmark = np.reshape(pre_landmark, (-1, 2))
        output_queue.put((frame, pre_landmark))


def run_onnx_model(queues, session, frame):
    for queue in queues:
        if not queue.full():
            queue.put(frame)
            break


class LEAP_C:
    def __init__(self, model):
        self.last_lid = None
        self.current_image_gray = None
        self.current_image_gray_clean = None
        onnxruntime.disable_telemetry_events()
        self.num_threads = 1
        self.queue_max_size = 1
        # self.model_path = "D:/EyeTrack_Break/PostProcess/Models/pfld-sim.onnx"

        self.print_fps = False
        self.frames = 0
        self.queues = [Queue(maxsize=self.queue_max_size) for _ in range(self.num_threads)]
        self.threads = []
        self.model_output = np.zeros((12, 2))
        self.output_queue = Queue(maxsize=self.queue_max_size)
        self.start_time = time.time()

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern = False

        self.one_euro_filter_float = OneEuroFilter(np.random.rand(1, 2), min_cutoff=0.0004, beta=0.9)
        self.dmax = 0
        self.dmin = 0
        self.openlist = []
        self.maxlist = []
        self.previous_time = None
        self.old_matrix = None
        self.total_velocity_new = 0
        self.total_velocity_avg = 0
        self.total_velocity_old = 0
        self.old_per = 0.0
        self.delta_per_neg = 0.0
        self.ort_session1 = onnxruntime.InferenceSession(self.model_path, opts, providers=["CPUExecutionProvider"])

        for i in range(self.num_threads):
            thread = threading.Thread(
                target=run_model,
                args=(self.queues[i], self.output_queue, self.ort_session1),
                name=f"Thread {i}",
            )
            self.threads.append(thread)
            thread.start()

    def leap_run(self):
        img = self.current_image_gray_clean.copy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_height, img_width = img.shape[:2]

        frame = cv2.resize(img, (112, 112))
        imgvis = self.current_image_gray.copy()
        run_onnx_model(self.queues, self.ort_session1, frame)

        if not self.output_queue.empty():
            frame, pre_landmark = self.output_queue.get()

            for point in pre_landmark:
                x, y = point
                x = int(x * img_width)
                y = int(y * img_height)
                cv2.circle(imgvis, (x, y), 3, (255, 255, 0), -1)
                cv2.circle(imgvis, (x, y), 1, (0, 0, 255), -1)

            d1 = math.dist(pre_landmark[1], pre_landmark[3])
            d2 = math.dist(pre_landmark[2], pre_landmark[4])
            d = (d1 + d2) / 2

         #   if len(self.openlist) > 0 and d >= np.percentile(self.openlist, 80) and len(self.openlist) < self.calibration_samples:
           #     self.maxlist.append(d)


            normal_open = np.percentile(self.openlist, 90) if len(self.openlist) >= 10 else 0.8

            if self.calib == 0:
                self.openlist = []

            if len(self.openlist) < self.calibration_samples:
                self.openlist.append(d)

            try:
                if len(self.openlist) > 0:
                    per = (d - normal_open) / (np.percentile(self.openlist, 2) - normal_open)
                    per = 1 - per
                    per = np.clip(per, 0.0, 1.0)
                else:
                    per = 0.8
            except:
                per = 0.8

            x = pre_landmark[6][0]
            y = pre_landmark[6][1]

            self.last_lid = per
            calib_array = np.array([per, per]).reshape(1, 2)
            per = self.one_euro_filter_float(calib_array)[0][0]

            if per <= 0.25:
                per = 0.0

            return imgvis, float(x), float(y), per

        imgvis = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return imgvis, 0, 0, 0


class External_Run_LEAP:
    def __init__(self):
        self.algo = LEAP_C()

    def run(self, current_image_gray, current_image_gray_clean, calib, calibration_samples):
        self.algo.current_image_gray = current_image_gray
        self.algo.current_image_gray_clean = current_image_gray_clean
        self.algo.calib = calib
        self.algo.calibration_samples = calibration_samples
        img, x, y, per = self.algo.leap_run()
        return img, x, y, per


def image_process(frame):
        frame = cv2.resize(frame, (192, 192))
        img_np = np.array(frame, dtype=np.float32) / 255.0  #归一
        gray_img = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2] #3RGB值
        
        gray_img = np.expand_dims(np.expand_dims(gray_img, axis=0), axis=0)
    
        return gray_img



def load_image(imt_path):
    if not os.path.exists(imt_path):
        print(f"Error: Path {imt_path} does not exist")
        exit(1)
        
    data = []
    # 使用opencv读取一组图片
    for file in os.listdir(imt_path):
        # 只处理常见的图片格式
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_path = os.path.join(imt_path, file)
            try:
                # 读取图片为灰度图
                img = cv2.imread(file_path)
                if img is not None:
                    data.append(img)
                else:
                    print(f"Failed to load image: {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    if data:
        # 将列表转换为numpy数组
        data = np.array(data)
        print(f"Successfully loaded {len(data)} images")
        print(f"Array shape: {data.shape}")
        return data
    else:
        print("No images were loaded")
    
        

if __name__ == "__main__":
    # model_path = "D:/EyeTrack_Break/PostProcess/Models/LEAP062120246epoch.onnx"
    model_path = "D:/EyeTrack_Break/PostProcess/Models/daddy230210.onnx"
    # 使用onnxruntime 加载模型
    session = onnxruntime.InferenceSession(model_path)
    
    # 读取图像
    # imt_path = os.path.join("EyeTrackData", "1", "Right", "Right")
    # data = load_image(imt_path)
    # img_Test = data[0]
    img_Test = cv2.imread("EyeTrackData/1/Right/Right/2024-12-23_21-22-12_1349.png")
    
    #旋转图像 逆时针90度
    img_Test = cv2.rotate(img_Test, cv2.ROTATE_90_CLOCKWISE)
    
    # 保存原始图像用于可视化
    img_vis = cv2.resize(img_Test.copy(), (224, 224))

    
    # 处理图像
    img_Test = image_process(img_Test)
    
    # 获取模型输入名称
    input_name = session.get_inputs()[0].name
    # 创建输入字典
    ort_inputs = {input_name: img_Test}
    
    # 运行推理
    test_mark = session.run(None, ort_inputs)
    test_mark = np.reshape(test_mark, (-1, 2))
    print(test_mark)

    img_vis = cv2.resize(img_vis, (224, 224))
    # 可视化并放大图像
    time_ = 0
    x,y = test_mark[6][0], test_mark[6][1]
    
    x = int(x * 224)
    y = int(y * 224)
    
    cv2.circle(img_vis, (x, y), 3, (255, 255, 0), -1)
    cv2.circle(img_vis, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow("img", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    

    
   #  cv2.imwrite("img_vis.png", img_vis)
    
    