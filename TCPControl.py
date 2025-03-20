import socket

class TCPSocket:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 添加 socket 选项，允许地址重用
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.control_flag = False
        self.save_flag = False
        try:
            self.server.bind((self.ip, self.port))
            print(f"Server listening on {self.ip}:{self.port}")
        except OSError as e:
            print(f"端口 {self.port} 已被占用，请检查是否有其他程序正在使用该端口")
            # 可以选择关闭之前的 socket
            self.server.close()
            raise e

        self.save_flag = False
    def send_command(self, command):
        self.server.send(command.encode())
        
    def close(self):
        try:
            self.server.close()
        except Exception as e:
            print(f"关闭 socket 时发生错误: {e}")
        
    def handle_message(self):
        try:
            # 开始监听连接
            self.server.listen(1)
            print(f"Server listening on {self.ip}:{self.port}")
            
            # 等待客户端连接
            client_socket, client_address = self.server.accept()
            print(f"Connection from {client_address}")
            
            try:
                while True:
                    
                    # 接收消息
                    data = client_socket.recv(1024)
                    if not data:
                        break
                        
                    # 解码并打印收到的消息
                    message = data.decode()
                    print(f"Received message: {message}")
                    self.handle_saving(message)
                    # 发送确认消息
                    response = "Message received"
                    client_socket.send(response.encode())
                    
            except Exception as e:
                print(f"Error handling message: {e}")
                
            finally:
                client_socket.close()
        except Exception as e:
            print(f"Error in handle_message: {e}")
            self.close()

    def handle_saving(self, msg):
        if (msg == "save" or msg == "start"):
            self.save_flag = True
            self.control_flag = True
        elif(msg =="stop" or msg =="close"):
            self.save_flag = False
            self.control_flag = True
        else:
            self.control_flag = False
    
    def get_control_flag(self):
        return self.control_flag
    
    def get_save_flag(self):
        return self.save_flag

    def reset_control_flag(self):
        self.control_flag = False


if __name__ == "__main__":
    tcp_control = TCPSocket("127.0.0.1", 6789)
