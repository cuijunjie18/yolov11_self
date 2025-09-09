import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import cv2
from PIL import Image

from ultralytics import YOLO

model = YOLO("runs/cpri_od/exp3/weights/best.pt")

# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True)  # Display preds. Accepts all YOLO predict arguments

# from PIL
img_path = "data/cpri2/images/train/M2T1A746N1004434524495175778.jpg"
# im1 = Image.open(img_path)
im1 = cv2.imread(img_path)
results = model.predict(source=im1, save=False)  # save plotted images


result = results[0]

# 获取类别
class_map = ['cpri', 'cpri_not_screw', 'cpri_screw', 'power', 'power_not_screw', 'power_screw']
cls = result.boxes.cls.to('cpu').numpy()  # Added .boxes here
print(cls)
print(type(cls))
print(len(cls))
print([class_map[int(i)] for i in cls])

# 获取绘制后的图像
# annotated_frame = result.plot()  # 返回numpy数组(BGR格式)
# cv2.imwrite("demo_pil.jpg", annotated_frame)