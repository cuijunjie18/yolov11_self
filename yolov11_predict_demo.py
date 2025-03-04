from ultralytics import YOLO

print("Predict begin!")

# 加载预训练的 YOLOv11n 模型
model = YOLO('my_install_models/wide_face/best.pt')

# model = YOLO('my_install_models/detection/yolo11n.pt')

print(model.task)

source = 'data/armors_data/unused_imgs/val/val/nred5407.jpg' #更改为自己的图片路径

# 运行推理，并附加参数
model.predict(source, save=True)