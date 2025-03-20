import cv2


def BLOB(self, max_len=500):
    """
    BLOB（Binary Large Object）检测函数
    用于检测图像中的大型二值对象，通常用于检测瞳孔
    参数:
        max_len: 最大处理长度，默认500
    返回:
        cx, cy: 检测到的斑点中心坐标
        larger_threshold: 处理后的二值图像
    """
    
    # 对灰度图像进行二值化处理
    # gui_threshold是阈值，低于阈值的像素设为0，高于阈值的设为255
    _, larger_threshold = cv2.threshold(
        self.current_image_gray,
        int(self.settings.gui_threshold),
        255,
        cv2.THRESH_BINARY,
    )
    
    try:
        # 在二值图像中查找所有轮廓
        # cv2.RETR_TREE: 检测所有轮廓并重建层次结构
        # cv2.CHAIN_APPROX_SIMPLE: 仅保存轮廓的拐点信息，压缩存储
        contours, _ = cv2.findContours(
            larger_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # 按轮廓面积从大到小排序
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        # 如果没有找到任何轮廓，抛出异常
        if len(contours) == 0:
            raise RuntimeError("No contours found for image")
    except:
        # 检测失败，增加失败计数
        self.failed = self.failed + 1
        pass

    # 获取图像尺寸
    rows, cols = larger_threshold.shape

    # 遍历所有检测到的轮廓
    for cnt in contours:
        # 获取轮廓的外接矩形
        # x,y 是矩形左上角坐标，w,h 是宽度和高度
        (x, y, w, h) = cv2.boundingRect(cnt)

        # 检查轮廓尺寸是否在允许范围内
        # gui_blob_minsize 和 gui_blob_maxsize 定义了可接受的最小和最大尺寸
        if (
            not self.settings.gui_blob_minsize <= h <= self.settings.gui_blob_maxsize
            or not self.settings.gui_blob_minsize <= w <= self.settings.gui_blob_maxsize
        ):
            continue

        # 计算轮廓中心点坐标
        cx = x + int(w / 2)
        cy = y + int(h / 2)

        # 在原图上绘制轮廓（用于可视化）
        cv2.drawContours(self.current_image_gray, [cnt], -1, (0, 0, 0), 3)
        # 绘制外接矩形（用于可视化）
        cv2.rectangle(self.current_image_gray, (x, y), (x + w, y + h), (0, 0, 0), 2)

        # 检测成功，重置失败计数
        self.failed = 0
        # 返回检测到的中心点坐标和处理后的图像
        return cx, cy, larger_threshold

    # 如果没有找到合适的轮廓，增加失败计数并返回默认值
    self.failed = self.failed + 1
    return 0, 0, larger_threshold
