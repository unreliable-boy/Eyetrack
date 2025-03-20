import cv2


def blob_detection(image_gray, gui_threshold, gui_blob_minsize=20, gui_blob_maxsize=200):
    """
    BLOB（Binary Large Object）检测函数
    用于检测图像中的大型二值对象，通常用于检测瞳孔

    参数:
        image_gray: 灰度图像（输入为单通道图像）
        gui_threshold: 二值化的阈值
        gui_blob_minsize: 轮廓的最小尺寸
        gui_blob_maxsize: 轮廓的最大尺寸

    返回:
        cx, cy: 检测到的斑点中心坐标
        larger_threshold: 处理后的二值图像
    """
    # 对灰度图像进行二值化处理
    _, larger_threshold = cv2.threshold(
        image_gray,
        gui_threshold,
        255,
        cv2.THRESH_BINARY,
    )
    try:
        # 在二值图像中查找所有轮廓
        contours, _ = cv2.findContours(
            larger_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # 按轮廓面积从大到小排序
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    # 如果没有找到任何轮廓，抛出异常
        if len(contours) == 0:
            raise RuntimeError("No contours found in the image")
    except:
        # 检测失败
        print("检测失败")
        return (0, 0), larger_threshold
    rows, cols = larger_threshold.shape

    # 遍历所有检测到的轮廓
    for cnt in contours:
        # 获取轮廓的外接矩形
        (x, y, w, h) = cv2.boundingRect(cnt)

        # 检查轮廓尺寸是否在允许范围内
        if (
            not gui_blob_minsize <= h <= gui_blob_maxsize
            or not gui_blob_minsize <= w <= gui_blob_maxsize
        ):
            continue

        # 计算轮廓中心点坐标
        cx = x + int(w / 2)
        cy = y + int(h / 2)

        # 在图像上绘制轮廓（用于可视化）
        cv2.drawContours(image_gray, [cnt], -1, (0, 0, 0), 3)
        # 绘制外接矩形（用于可视化）
        cv2.rectangle(image_gray, (x, y), (x + w, y + h), (0, 0, 0), 2)

        # 返回检测到的中心点坐标和处理后的图像
        return (cx, cy), larger_threshold, image_gray

    # 如果没有找到合适的轮廓，返回默认值
    return (0, 0), larger_threshold

if __name__ == "__main__":
    img_path = "EyeTrackData/1/2025-02-24_16-15-56_5.png"
    image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    location, larger_threshold, image_gray = blob_detection(image_gray, 127)
    print(location)
    cv2.imshow("image_gray", image_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()