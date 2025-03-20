import os.path
import json
from typing import Union



class Config:
    """
    相机配置
    """
    def __init__(self,widget_id:int):
        self.widget_id = widget_id
        self.config = {}
        self.CONFIG_FILE_NAME = "settings.json"
        self.load_config()
        
       
        
    def load_config(self):
        
        try:
            if os.path.exists(self.CONFIG_FILE_NAME):
                with open(self.CONFIG_FILE_NAME, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    # 先获取根级别的配置
                    self.project_root = config_data.get('project_root')
                    self.DataNumber = config_data.get('DataNumber')
                    self.tcp_ip = config_data.get('tcp_ip')      # 从根级别获取 tcp_ip
                    self.tcp_port = config_data.get('tcp_port')  # 从根级别获取 tcp_port
                    
                    # 获取特定 widget 的配置
                    self.config = config_data.get(f'widget{self.widget_id}', {})
                    self.capture_source = self.config.get('capture_source')
                    self.image_save_address = self.config.get('image_save_address')
                    
                    print(f"TCP Control: {self.tcp_ip}:{self.tcp_port}")
            else:
                print(f"[WARN] 配置文件 {self.CONFIG_FILE_NAME} 不存在")
                self.create_default_config(self.widget_id)
        except json.JSONDecodeError as e:
            print(f"[ERROR] 配置文件格式错误: {str(e)}")
        except Exception as e:
            print(f"[ERROR] 加载配置文件失败: {str(e)}")
            

    def save_config(self,dataname,data):
        """保存配置到文件"""
        try:
            # 先读取现有配置（如果存在）
            if os.path.exists(self.CONFIG_FILE_NAME):
                with open(self.CONFIG_FILE_NAME, 'r', encoding='utf-8') as f:
                    try:
                        self.config = json.load(f)
                    except json.JSONDecodeError:
                        pass
            self.config[f'widget{self.widget_id}'][dataname] = data
            
            # 保存整个配置文件
            with open(self.CONFIG_FILE_NAME, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
            print(f"[INFO] Widget {self.widget_id} 配置已保存")
        except Exception as e:
            print(f"[ERROR] 保存配置文件失败: {str(e)}")

    def update_port(self, widget_id: int, new_port: str):
        """更新特定widget的端口配置"""
        self.capture_source = new_port
        self.save_config(widget_id)


if __name__ == "__main__":
    # 测试代码
    config = Config(0)


    

