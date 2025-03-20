import cv2
import numpy as np
import queue
import serial
import serial.tools.list_ports
import threading
import time
import json
from enum import Enum
from colorama import Fore, Style
import psutil, os
import sys
from config import Config



process = psutil.Process(os.getpid())
try:
    sys.getwindowsversion()
except AttributeError:
    process.nice(10)  # UNIX: 0 low 10 high
    process.nice()
else:
    process.nice(psutil.HIGH_PRIORITY_CLASS)  # Windows
    process.nice()
    
class CameraState(Enum):
    CONNECTING = 0
    CONNECTED = 1
    DISCONNECTED = 2

WAIT_TIME = 0.1

ETVR_HEADER = b"\xff\xa0"
ETVR_HEADER_FRAME = b"\xff\xa1"
ETVR_HEADER_LEN = 6


def is_serial_capture_source(addr: str) -> bool:
    """
    Returns True if the capture source address is a serial port.
    """
    return (
        addr.startswith("COM") or addr.startswith("/dev/cu") or addr.startswith("/dev/tty")  # Windows  # macOS  # Linux
    )

current_capture_source = "COM4"

class Camera:
    def __init__(
            self, 
            config: Config,
            widget_id: int,
            cancellation_event: "threading.Event",
            capture_event: "threading.Event",
            camera_output_outgoing: "queue.Queue",
    ):
        self.config = config
        self.current_capture_source = self.config.capture_source
        self.serial_connection = None
        self.camera_output_outgoing = camera_output_outgoing
        self.buffer = b""
        self.capture_event = capture_event
        self.camera_status = CameraState.CONNECTING
        self.cancellation_event = cancellation_event

        self.frame_number = 0

        self.fail_connect_count = 0
        self.img = None

    def __del__(self):
        if self.serial_connection is not None:
            self.serial_connection.close()

    def run(self):
        
        while True:
            if self.cancellation_event.is_set():
                print(f"{Fore.CYAN}[INFO] Exiting Capture thread{Fore.RESET}")
                return
            should_push = True
            self.check_config_port()
            port = self.current_capture_source
            
            if (self.camera_status == CameraState.DISCONNECTED or 
                self.camera_status == CameraState.CONNECTING):
                self.start_serial_connection(port)
            """
            else:
                # We don't have a capture source to try yet, wait for one to show up in the GUI.
                if self.cancellation_event.wait(WAIT_TIME):
                    self.camera_status = CameraState.DISCONNECTED
                    return
            """
            if self.camera_status == CameraState.CONNECTED:
                addr = str(self.current_capture_source)
                if is_serial_capture_source(addr):
                    self.get_serial_camera_picture(should_push)
                    # image = self.camera_output_outgoing.get(block=False)
                    # print(f"frame_number: {self.frame_number}")
                    

                if not should_push:
                    # if we get all the way down here, consider ourselves connected
                    self.camera_status = CameraState.CONNECTED
                    
                    
    def set_output_queue(self, camera_output_outgoing: "queue.Queue"):
        self.camera_output_outgoing = camera_output_outgoing

    def get_next_packet_bounds(self):
        beg = -1
        while beg == -1:
            self.buffer += self.serial_connection.read(2048)
            beg = self.buffer.find(ETVR_HEADER + ETVR_HEADER_FRAME)
        # Discard any data before the frame header.
        if beg > 0:
            self.buffer = self.buffer[beg:]
            beg = 0
        # We know exactly how long the jpeg packet is
        end = int.from_bytes(self.buffer[4:6], signed=False, byteorder="little")
        self.buffer += self.serial_connection.read(end - len(self.buffer))
        return beg, end

    def get_next_jpeg_frame(self):
        beg, end = self.get_next_packet_bounds()
        jpeg = self.buffer[beg + ETVR_HEADER_LEN: end + ETVR_HEADER_LEN]
        self.buffer = self.buffer[end + ETVR_HEADER_LEN:]
        return jpeg

    def get_serial_camera_picture(self, should_push):
        conn = self.serial_connection
        if conn is None:
            return

        current_frame_time = time.time()
        self.prevtime = current_frame_time
        
        try:
            if conn.in_waiting:
                jpeg = self.get_next_jpeg_frame()
                if jpeg: 
                    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    if image is None:
                        print(f"{Fore.YELLOW}[WARN] Frame drop. Corrupted JPEG.{Fore.RESET}")
                        return
                    if conn.in_waiting >= 32768:
                        print(f"{Fore.CYAN}[INFO] Discarding the serial buffer ({conn.in_waiting} bytes){Fore.RESET}")
                        conn.reset_input_buffer()
                        self.buffer = b""
                    self.frame_number += 1
                    
                    if should_push and image is not None:
                        self.push_image_to_queue(image, self.frame_number)
                    
        except serial.SerialException as se:
            print(f"{Fore.RED}[ERROR] 串口通信错误: {str(se)}{Fore.RESET}")
            conn.close()
            self.camera_status = CameraState.DISCONNECTED
        except Exception as e:
            print(f"{Fore.RED}[ERROR] 未知错误: {str(e)}{Fore.RESET}")
            print(f"{Fore.YELLOW}[WARN] 串口捕获源出现问题，假定相机已断开，等待重新连接。{Fore.RESET}")
            conn.close()
            self.camera_status = CameraState.DISCONNECTED
    
    def set_output_queue(self, camera_output_outgoing: "queue.Queue"):
        self.camera_output_outgoing = camera_output_outgoing 
        
    def start_serial_connection(self, port):
        com_ports = [tuple(p) for p in list(serial.tools.list_ports.comports())]
        if not any(p for p in com_ports if port in p):
            print(f"{Fore.CYAN}[INFO] No Device on {port}{Fore.RESET}")
            time.sleep(1)
            self.fail_connect_count += 1
            return
        try:
            rate = 115200 if sys.platform == "darwin" else 3000000  # Higher baud rate not working on macOS
            conn = serial.Serial(baudrate=rate, port=port, xonxoff=False, dsrdtr=False, rtscts=False)
            if sys.platform == "win32":
                buffer_size = 32768
                conn.set_buffer_size(rx_size=buffer_size, tx_size=buffer_size)
            print(f"{Fore.CYAN}[INFO] Serial Tracker device connected on {port}{Fore.RESET}")
            self.serial_connection = conn
            self.camera_status = CameraState.CONNECTED
        except Exception as e:
            print(f"{Fore.CYAN}[INFO] Failed to connect on {port}{Fore.RESET}")
            print(f"{Fore.RED}Error: {e}{Fore.RESET}")
            self.camera_status = CameraState.DISCONNECTED

    def push_image_to_queue(self, image, frame_number):
        # If there's backpressure, just yell. We really shouldn't have this unless we start getting
        # some sort of capture event conflict though.
        qsize = self.camera_output_outgoing.qsize()
        if qsize > 1:
            print(
                f"{Fore.YELLOW}[WARN] CAPTURE QUEUE BACKPRESSURE OF {qsize}. CHECK FOR CRASH OR TIMING ISSUES IN ALGORITHM.{Fore.RESET}"
            )

        self.camera_output_outgoing.put((image, frame_number))
        
    def check_config_port(self):
        """检查配置文件中的端口是否与当前端口一致"""
        try:
            if self.config.capture_source and self.config.capture_source != self.current_capture_source:
                print(f"{Fore.YELLOW}[INFO] 从配置文件更新端口: {self.current_capture_source} -> {self.config.capture_source}{Fore.RESET}")
                self.current_capture_source = self.config.capture_source
                # 如果已经连接，需要重新连接
                if self.serial_connection is not None:
                    self.serial_connection.close()
                    self.camera_status = CameraState.DISCONNECTED
        except Exception as e:
            print(f"{Fore.RED}[ERROR] 读取配置文件失败: {str(e)}{Fore.RESET}")

if __name__ == "__main__":
    config = Config()
    cam = Camera(config,threading.Event(),threading.Event(),queue.Queue())
    cam.run()
