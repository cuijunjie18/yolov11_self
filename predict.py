import cv2
from PIL import Image

from ultralytics import YOLO

model = YOLO("runs/obb/train/exp3/weights/best.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True)  # Display preds. Accepts all YOLO predict arguments

# # from PIL
# im1 = Image.open("bus.jpg")
# results = model.predict(source=im1, save=True)  # save plotted images

# from ndarray
img_path = "/data_all/cjj_node/HuaWei_data_annotaion/Huawei/BBU接地质量/接地线标签/不合格/M1T1A746N817823465499385932.jpg"
im2 = cv2.imread(img_path)
results = model.predict(source=im2, save=False, save_txt=False)  # save predictions as labels

# 假设输入单张图片，取第一个结果
result = results[0]

obb = result.obb

xyxys = obb.xyxy

for xyxy in xyxys:
    xmin,ymin,xmax,ymax = xyxy.to('cpu')
    xmin = int(xmin.item())
    ymin = int(ymin.item())
    xmax = int(xmax.item())
    ymax = int(ymax.item())
    cv2.rectangle(im2,(xmin,ymin),(xmax,ymax),color = (0,0,255),thickness = 3)
cv2.imwrite("demo.jpg",im2)