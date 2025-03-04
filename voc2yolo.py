import os
import xml.etree.ElementTree as ET

# 类别列表，根据你的数据集修改
classes = ["cat", "dog"]    # 这里的类别索引要记住，0代表cat，1代表dog

def convert_voc_to_yolo(xml_file, output_dir):
    # 解析 XML 文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 获取图像尺寸
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    # 准备输出文件路径
    image_name = root.find("filename").text
    txt_file = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".txt")

    with open(txt_file, "w") as f:
        # 遍历每个目标
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in classes:
                continue  # 忽略未定义的类别

            # 获取类别索引
            class_id = classes.index(class_name)

            # 获取目标框坐标
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            # 转换为 YOLO 格式
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            # 写入文件
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# # 示例：转换单个文件
# xml_file = "path/to/your/annotation.xml"
# output_dir = "path/to/output/dir"
# # convert_voc_to_yolo(xml_file, output_dir)

# 示例：批量转换整个目录
def convert_voc_dir_to_yolo(xml_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith(".xml"):
            convert_voc_to_yolo(os.path.join(xml_dir, xml_file), output_dir)

xml_dir = "data/animals_data/Annotations"
output_dir = "data/animals_data/labels"
convert_voc_dir_to_yolo(xml_dir, output_dir)