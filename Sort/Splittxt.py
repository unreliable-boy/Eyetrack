import os

def split_txt_file(input_file, output_dir, slice_size=20000):
    """
    将一个长文本文件按照指定大小切片，并删除空行。

    参数:
        input_file (str): 输入的长文本文件路径
        output_dir (str): 输出切片文件的目录
        slice_size (int): 每个切片文件的大小（默认20000字）
    """
    import os

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取输入文件并处理空行
    with open(input_file, 'r', encoding='utf-8') as f:
        # 读取所有行并删除空行
        lines = [line.strip() for line in f if line.strip()]

    # 将处理后的文本合并为一个字符串
    text = '\n'.join(lines)

    # 计算总字数和需要的切片数量
    total_chars = len(text)
    num_slices = (total_chars + slice_size - 1) // slice_size  # 向上取整

    # 按照两万字一个切片分割文本
    for i in range(num_slices):
        start = i * slice_size
        end = start + slice_size
        slice_text = text[start:end]

        # 生成输出文件名（从001开始）
        file_name = f"{i+1:03d}.txt"
        output_path = os.path.join(output_dir, file_name)

        # 写入切片文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(slice_text)

        print(f"已生成文件: {file_name}")

# 使用示例
input_file = 'Sort/LongZu.txt'  # 输入的长文本文件路径
output_dir = 'Sort/sliced_files'   # 输出切片文件的目录

split_txt_file(input_file, output_dir)