import msgpack
import os
import numpy as np

def load_refraction_model(model_name="default_refraction_model_radius_degree_3.msgpack"):
    # 获取模型文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, model_name)
    
    # 读取模型参数
    try:
        with open(model_path, 'rb') as f:
            model_params = msgpack.unpack(f)
            
        # 转换为numpy数组（如果需要）
        if isinstance(model_params, dict):
            for key in model_params:
                if isinstance(model_params[key], list):
                    model_params[key] = np.array(model_params[key])
                    
        return model_params
        
    except FileNotFoundError:
        print(f"找不到模型文件: {model_name}")
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        
    return None

# 使用示例
model = load_refraction_model()
if model:
    print("模型参数:")
    for key, value in model.items():
        print(f"{key}:")
        print(value)

