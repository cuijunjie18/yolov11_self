from ultralytics import YOLO
from torchsummary import summary
import torchvision
import torch

# 加载模型
model = YOLO('my_install_models/wide_face/best.pt')

model.export(format = 'onnx')