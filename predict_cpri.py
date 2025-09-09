import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import cv2
from PIL import Image

from ultralytics import YOLO

model = YOLO("runs/cpri_od/exp3/weights/best.pt")

# from ndarray
img_path = "data/cpri2/images/train/M2T1A746N1004434524495175778.jpg"
im2 = cv2.imread(img_path)
results = model.predict(source=im2, save=False, save_txt=False)  # save predictions as labels

# 假设输入单张图片，取第一个结果
result = results[0]

xyxys = result.boxes.xyxy  # Added .boxes here

for xyxy in xyxys:
    xmin,ymin,xmax,ymax = xyxy.to('cpu')
    xmin = int(xmin.item())
    ymin = int(ymin.item())
    xmax = int(xmax.item())
    ymax = int(ymax.item())
    cv2.rectangle(im2,(xmin,ymin),(xmax,ymax),color = (0,0,255),thickness = 3)
cv2.imwrite("demo.jpg",im2)