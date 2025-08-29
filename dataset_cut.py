import os
import random
import shutil

def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # 创建输出目录
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "test"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "test"), exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")] # 注意这里是jpg
    random.shuffle(image_files)

    # 计算划分点
    num_images = len(image_files)
    train_end = int(num_images * train_ratio)
    val_end = train_end + int(num_images * val_ratio)

    # 划分数据集
    for i, image_file in enumerate(image_files):
        label_file = os.path.splitext(image_file)[0] + ".txt"
        if i < train_end:
            subset = "train"
        elif i < val_end:
            subset = "val"
        else:
            subset = "test"

        # 复制图像和标签文件
        shutil.copy(os.path.join(image_dir, image_file), os.path.join(output_dir, "images", subset, image_file))
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(output_dir, "labels", subset, label_file))

# 示例：划分数据集
image_dir = "data/CPRI_coco_dataset/raw_images"
label_dir = "data/CPRI_coco_dataset/raw_labels"
output_dir = "data/CPRI_coco_dataset"
split_dataset(image_dir, label_dir, output_dir)