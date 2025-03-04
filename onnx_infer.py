import torch.nn as nn
import torch
import numpy as np
import onnx
import onnxruntime
import cv2

# letterbox 填充
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    """
    将图像进行 letterbox 填充，保持纵横比不变，并缩放到指定尺寸。
    """
    shape = img.shape[:2]  # 当前图像的宽高 (h,w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 选择宽高中最小的缩放比
    if not scaleup:  # 仅缩小，不放大
        r = min(r, 1.0)

    # 缩放后的未填充尺寸
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))   # (new_w,new_h)

    # 计算需要的填充
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算填充的尺寸
    dw /= 2  # padding 均分
    dh /= 2

    # 缩放图像
    if shape[::-1] != new_unpad:  # 如果当前图像尺寸不等于 new_unpad，则缩放
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 为图像添加边框以达到目标尺寸
    top, bottom = int(round(dh)), int(round(dh))
    left, right = int(round(dw)), int(round(dw))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (r, r), (dw, dh)

# 图片预处理
def preprocess(img_path,input_info):
    """
    输入 :
    img_path : 图片路径
    input_info : 模型输入层的信息
    输出 :
    output_data : 符合模型输入的数据(numpy格式)
    ratio : 缩放比例
    (dw,dh) : 填充尺寸
    """

    # 法一：np格式的处理方式
    input_h,input_w = input_info.shape[2:]      # 获取模型输入大小
    img = cv2.imread(img_path)                  # BGR格式读入
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   # 转化为RGB格式

    # 使用letterbox 方法保持宽高比
    img,ratio,(dw,dh) = letterbox(img,new_shape = (input_w,input_h))

    print(f"padding dw,dh = {(dw,dh)}")

    # img = cv2.resize(img,(input_w,input_h))     # 调整大小
    img = np.array(img,dtype = np.float32)      # 浮点转化
    img /= 255.0                                # 图像归一化
    img = np.transpose(img,(2,0,1))             # 颜色通道优先
    img = np.expand_dims(img,axis = 0)          # 添加批大小：1
    # print(img.shape)                          # 检查格式

    # 法二：torch格式的处理
    # img = cv2.resize(cv2.imread(img_path),(640,640)) # 调整大小
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # BGR2RGB
    # x = torch.tensor(img,dtype = torch.float32) # 转化为tensor
    # x /= 255.0 # 归一化
    # x = x.permute(2,0,1).unsqueeze(0) #  转化成yolo模型的输入格式
    # print(type(x),x.shape)
    return img, ratio, (dw, dh)   

# 输出结果后处理
def postprocess(output, ratio, dw, dh, conf_threshold=0.5, iou_threshold=0.5):
    """
    输入 :
    output : 模型推理输出结果
    ratio : 缩放比例
    (dw,dh) : 填充尺寸
    输出 :
    boxes : 边界框坐标 (x_min, y_min, x_max, y_max)
    scores : 置信度分数
    class_ids : 类别 ID
    """
    # 输出形状: (1, 84, 8400)
    output = output[0]  # 去掉批量维度 -> (84, 8400)

    # 提取边界框和类别分数
    boxes = output[:4, :]  # (4, 8400): x_center, y_center, width, height
    scores = output[4:, :]  # (80, 8400): 类别概率分数

    # 找到每个预测框的最大类别分数
    max_scores = np.max(scores, axis=0)  # (8400,)
    max_class_ids = np.argmax(scores, axis=0)  # (8400,)

    # 过滤低置信度的预测框
    keep = max_scores > conf_threshold
    boxes = boxes[:, keep]
    scores = max_scores[keep]
    class_ids = max_class_ids[keep]

    # 将边界框从 (x_center, y_center, width, height) 转换为 (x_min, y_min, x_max, y_max)
    x_min = boxes[0, :] - boxes[2, :] / 2   # (n,) n为过滤低置信度后的预测框数量
    y_min = boxes[1, :] - boxes[3, :] / 2
    x_max = boxes[0, :] + boxes[2, :] / 2
    y_max = boxes[1, :] + boxes[3, :] / 2
    boxes = np.stack([x_min, y_min, x_max, y_max], axis=1) # (n,4)

    # # 非极大值抑制 (NMS)
    # keep_indices = nms(boxes, scores, iou_threshold)
    # boxes = boxes[keep_indices]
    # scores = scores[keep_indices]
    # class_ids = class_ids[keep_indices]

    # return boxes, scores, class_ids

    # 调整边界框坐标，考虑缩放和填充
    boxes[:, [0, 2]] -= dw  # 移除填充
    boxes[:, [1, 3]] -= dh
    boxes[:, [0, 2]] /= ratio[0]  # 缩放回原图
    boxes[:, [1, 3]] /= ratio[1]

    print(f"Before nms boxes shape : {boxes.shape}")

    # 使用 OpenCV 的 NMSBoxes 进行非极大值抑制
    boxes_xywh = np.zeros_like(boxes)
    boxes_xywh[:, 0] = boxes[:, 0]  # x_min
    boxes_xywh[:, 1] = boxes[:, 1]  # y_min
    boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
    boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height

    keep_indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    if len(keep_indices) > 0:
        boxes = boxes[keep_indices.flatten()]
        scores = scores[keep_indices.flatten()]
        class_ids = class_ids[keep_indices.flatten()]

    return boxes, scores, class_ids

# nms非极大值抑制
def nms(boxes, scores, iou_threshold):
    # 非极大值抑制实现
    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    x_max = boxes[:, 2]
    y_max = boxes[:, 3]

    areas = (x_max - x_min) * (y_max - y_min)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x_min[i], x_min[order[1:]])
        yy1 = np.maximum(y_min[i], y_min[order[1:]])
        xx2 = np.minimum(x_max[i], x_max[order[1:]])
        yy2 = np.minimum(y_max[i], y_max[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

# 可视化结果
def visualize(image_path, input_info,boxes, scores, class_ids, class_names):
    """
    输入
    image_path : 图片路径
    input_info : 模型输入信息
    boxes : 预测框(在模型输入尺度上的)
    scores : 预测框置信度
    class_ids : 类型下标
    class_names : 类型下标到类型字符串名字的映射
    """
    input_h,input_w = input_info.shape[2:]      # 获取模型输入大小
    image = cv2.imread(image_path)
    for box, score, class_id in zip(boxes, scores, class_ids):
        x_min, y_min, x_max, y_max = box
        label = f"{class_names[class_id]} {score:.2f}"
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 1)
        cv2.putText(image, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    # 检查模型是否存在
    onnx_file_path = "my_install_models/wide_face/best.onnx" # 模型所在位置
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)

    # 初始化ONNX模型
    ort_session = onnxruntime.InferenceSession(onnx_file_path)

    # 获取模型输入信息
    input_info = ort_session.get_inputs()
    print(f"Model input shape:{input_info[0].shape}")
    input_info = input_info[0]

    # 预处理待detect的图片
    img_path = 'data/armors_data/unused_imgs/val/val/nred5407.jpg'
    input_data, ratio, (dw, dh) = preprocess(img_path,input_info)

    # 为onnx模型构造输入
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}

    # ONNX模型推理
    ort_outs = ort_session.run(None, ort_inputs)

    # # 可视化输出类型及形状
    # print(type(ort_outs))
    # print(len(ort_outs))
    # print(type(ort_outs[0]))
    # print(ort_outs[0].shape)

    # 示例：后处理输出
    boxes, scores, class_ids = postprocess(ort_outs[0],ratio, dw, dh)
    print(f"After nms boxes shape : {boxes.shape}")
    print("Boxes:", boxes)
    print("Scores:", scores)
    print("Class IDs:", class_ids)

    # 示例：可视化结果
    # class_names = ["class1", "class2", ..., "class80"]  # 替换为你的类别名称
    class_names = []
    for i in range(100):
        class_names.append(str(i))
    visualize(img_path,input_info, boxes, scores, class_ids, class_names)


