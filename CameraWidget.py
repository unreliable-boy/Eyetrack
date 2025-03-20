import PySimpleGUI as sg
import cv2
import os
from threading import Event, Thread
from enum import Enum
from queue import Queue, Empty
import time
import datetime
from Camear import Camera
from config import Config


class CameraWidget:
    def __init__(self, widget_id:int):
         # Initialize synchronization primitives and queue
        self.cancellation_event = Event()
        self.capture_event = Event()
        self.capture_queue = Queue()
        self.config = Config(widget_id)

        self.last_render_time = time.time()
        self.currentimage = None
        self.save_frame_count = 1
        self.widget_id = widget_id

        
        # Instantiate the Camera
        self.camera = Camera(
            config=self.config,
            widget_id=self.widget_id,
            cancellation_event=self.cancellation_event,
            capture_event=self.capture_event,
            camera_output_outgoing=self.capture_queue
        )
        self.capture_source = self.config.capture_source
        self.init_save_dir()

        # Instantiate the GUI Key
        self.gui_widget_layout = f"-WIGHTLAYOUT{widget_id}-"
        self.gui_tracking_layout = f"-TRACKINGLAYOUT{widget_id}-"
        self.gui_graph = f"-GRAPH{widget_id}-"
        self.gui_port = f"-PORT{widget_id}-"
        self.gui_save_address = f"-SAVE_ADDRESS{widget_id}-"
        self.gui_address_layout = f"-ADRLAYOUT{widget_id}-"
        self.gui_savedir = f"-SAVEDIR{widget_id}-"
        self.gui_savedir_layout = f"-SAVEDIRLAYOUT{widget_id}-"
        self.gui_fps = f"-FPS{widget_id}-"
        self.gui_frame = f"-FRAME{widget_id}-"
        self.gui_status = f"-STATUS{widget_id}-"
        self.get_tracking_layout()
        self.get_address_layout()
        self.get_savedir_layout()
        self.tcp_ip = self.config.tcp_ip
        self.tcp_port = self.config.tcp_port
        

    def started(self):
        return not self.cancellation_event.is_set()

    def start_camera(self):
        print("[DEBUG] 准备启动相机线程...")
        try:
            self.camera_thread = Thread(
                target=self.camera.run,
                name="CameraThread",
                daemon=True  
            )
            print(f"[DEBUG] 相机线程创建成功: {self.camera_thread.name}")
            
            if not self.camera_thread.is_alive():
                print("[DEBUG] 开始启动相机线程...")
                self.cancellation_event.clear()  # 确保清除取消标志
                self.capture_event.set()  # 确保设置捕获标志
                self.camera_thread.start()
                print(f"[DEBUG] 相机线程状态: {'活跃' if self.camera_thread.is_alive() else '未活跃'}")
            else:
                print("[WARNING] 相机线程已经在运行中")
        except Exception as e:
            print(f"[ERROR] 启动相机线程时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def stop_camera(self):
        if self.camera_thread.is_alive():
            print("[INFO] Stopping camera thread.")
            self.cancellation_event.set()
            self.camera_thread.join()  # Wait for the thread to finish
            print("[INFO] Camera thread stopped.")
        else:
            print("[WARNING] Camera thread is not running.")


    def get_widget_layout(self):
        self.widget_layout = [
                [sg.Column(
                    self.address_layout,
                    key=self.gui_address_layout,
                        background_color="#424042"
                    )
                ],
                [sg.Column(
                    self.savedir_layout,
                    key=self.gui_savedir_layout,
                    background_color="#424042"
                )],
                [sg.Column(
                    self.tracking_layout,  
                    key=self.gui_widget_layout,
                    background_color="#424042"
                )]
        ]
        return self.widget_layout

    def get_status_layout(self):
        self.status_layout = [
            [sg.Text(f"Status: {self.Status}", key=self.gui_status)]
        ]
        return self.status_layout

    def get_tracking_layout(self):
        self.tracking_layout = [
            [sg.Graph(
                canvas_size=(200, 200),      
                graph_bottom_left=(-100, -100),  
                graph_top_right=(100, 100),   
                background_color='black',     
                key=self.gui_graph,
                enable_events=True           
            )],
            [
                sg.Text(f"FPS: 0", key=self.gui_fps),
                sg.Text(f"Frame: 0", key=self.gui_frame)
            ]
        ]

    def get_address_layout(self):
        self.address_layout = [
            [sg.Text('Capture Source:'), 
             sg.Input(default_text=self.capture_source, key=self.gui_port, size=(10,1)),
             sg.Button("Save", key=self.gui_save_address)]
        ]
        
    
    def get_savedir_layout(self):
        self.savedir_layout = [
            [sg.Text('Save Directory:'), 
             sg.Text(self.DataNumber + self.config.image_save_address)]
        ]


    def save_image(self, image, frame_number):
        """保存图像到指定目录"""
        current_time = datetime.datetime.now()
        try:
            # 确保保存目录存在
            os.makedirs(self.save_dir, exist_ok=True)
            # 使用os.path.join构建完整的文件路径

            save_name = current_time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{frame_number}.png"
            
            save_path = os.path.join(self.save_dir, save_name)
            
            # 保存图像
            cv2.imwrite(save_path, image)
            print(f"[DEBUG] 保存图像: {save_path}")
        except Exception as e:
            pass

    def init_save_dir(self):
        self.project_root = self.config.project_root
        self.DataNumber = self.config.DataNumber
        self.save_dir = self.project_root + self.DataNumber + self.config.image_save_address    
        print(f"[DEBUG] 保存目录: {self.save_dir}")

    def render(self, window, event, value):
        current_time = time.time()
        elapsed_time = current_time - self.last_render_time

        if event == self.gui_address_layout:
            self.capture_source = value[self.gui_port]
            self.config.save_config('capture_source',value[self.gui_port])
            self.camera.check_config_port()
            print(f"[DEBUG] 保存配置: {self.capture_source}")
            
        self.camera.set_output_queue(self.capture_queue)
        try:
            self.currentimage, self.frame_number = self.capture_queue.get(block=False)
            self.last_render_time = current_time  # 更新最后渲染时间
            imgbytes = cv2.imencode('.ppm', self.currentimage)[1].tobytes()
            graph = window[self.gui_graph]
            graph.erase()
            graph.draw_image(data=imgbytes, location=(-100, 100))
            # 显示当前帧率
            fps = 1.0/elapsed_time
            window[self.gui_fps].update(f"FPS: {fps:.1f}")
            window[self.gui_frame].update(f"Frame: {self.frame_number}")
            
        except Empty:
            pass
        except Exception as e:
            print(f"[ERROR] 渲染错误: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def save_image_event(self, window, event, values):
        try:
            self.save_image(self.currentimage, self.save_frame_count)
            self.save_frame_count += 1
            
        except Empty:
            print("[DEBUG] 没有可保存的图像")
        except Exception as e:
            # print(f"[ERROR] 保存图像错误")
            # import traceback
            # print(traceback.format_exc())
            pass

def main():
    camera_widget = CameraWidget(0)
    
    layout = camera_widget.get_widget_layout()
    window = sg.Window("Tracking UI", layout=layout, finalize=True)

    while True:
        event, values = window.read(timeout=1)
        if event == sg.WIN_CLOSED:
            if camera_widget.started():  # 确保在关闭窗口时停止相机
                camera_widget.stop_camera()
            print("退出程序")
            return
        
        if event == f'-START_CAMERA{camera_widget.widget_id}-':
            camera_widget.start_camera()
            window[f'-START_CAMERA{camera_widget.widget_id}-'].update(disabled=True)  # 禁用启动按钮
            window[f'-STOP_CAMERA{camera_widget.widget_id}-'].update(disabled=False)  # 启用停止按钮

        elif event == f'-STOP_CAMERA{camera_widget.widget_id}-':
            camera_widget.stop_camera()
            window[f'-START_CAMERA{camera_widget.widget_id}-'].update(disabled=False)  # 启用启动按钮
            window[f'-STOP_CAMERA{camera_widget.widget_id}-'].update(disabled=True)   # 禁用停止按钮


        if camera_widget.started():
            camera_widget.render(window, event, values)
    
    


if __name__ == "__main__":
    main()
    