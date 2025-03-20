import PySimpleGUI as sg
from Camear import Camera
from CameraWidget import CameraWidget
from enum import Enum
from TCPControl import TCPSocket
from config import Config
import threading

class Eye(Enum):
    LEFT = 0
    RIGHT = 1 

config = Config(0)
ip = config.tcp_ip
port = config.tcp_port

def get_status_layout():
    status_layout = [
        [  
            sg.Button("Start Camera", key='-START_CAMERA-'),
            sg.Button("Stop Camera", key='-STOP_CAMERA-'),
            sg.Button("Start Saving Image", key='-SAVE_IMAGE-'),
            sg.Button("Stop Saving Image", key='-STOP_SAVE_IMAGE-'),
            sg.Text("By ZhaoKeao", key='-Signature-')
        ],
        [
            sg.Text(f"ServerIP: {ip}:{port}", key='-TCP_CONTROL-'),
        ]
    ]
    return status_layout


    


def main():
    TCP = TCPSocket(ip, port)
    tcp_thread = threading.Thread(target=TCP.handle_message, daemon=True)
    tcp_thread.start()
    
    Save_Flag = False
    Button_Control = False  # 新增按钮控制标志
    eyes = {
        Eye.LEFT: CameraWidget(0),  # 左眼
        Eye.RIGHT: CameraWidget(1)  # 右眼
    }
    
   
    status_layout = get_status_layout()
    layout = [
        [sg.Column(eyes[Eye.LEFT].get_widget_layout(), background_color="#424042"),
         sg.VSeparator(),  # 添加垂直分隔线
         sg.Column(eyes[Eye.RIGHT].get_widget_layout(), background_color="#424042")],
        [sg.Column(status_layout, background_color="#424042")]
    ]

    window = sg.Window("Eye Tracking UI", layout=layout, finalize=True)

    while True:
        event, values = window.read(timeout=1)
        TCP_Control = TCP.get_control_flag()
        
        # 处理TCP控制
        if TCP_Control:
            Save_Flag = TCP.get_save_flag()
        
        # 处理按钮控制
        if event == f'-SAVE_IMAGE-':
            Button_Control = True
            Save_Flag = True
        elif event == f'-STOP_SAVE_IMAGE-':
            Button_Control = True
            Save_Flag = False
            
        # 如果是按钮控制，则忽略TCP控制
        if Button_Control:
            TCP.reset_control_flag()  # 重置TCP控制标志
        
        if event == sg.WIN_CLOSED:
            for eye in eyes.values():
                if eye.started():
                    eye.stop_camera()
            print("退出程序")
            return
        
        if event == f'-START_CAMERA-':
            for eye in eyes.values():
                eye.start_camera()
                
        elif event == f'-STOP_CAMERA-':
            for eye in eyes.values():
                eye.stop_camera()
                
        for eye in eyes.values():
            # 渲染画面
            if eye.started():
                eye.render(window, event, values)
                if Save_Flag:
                    eye.save_image_event(window, event, values)



if __name__ == "__main__":
    main()
