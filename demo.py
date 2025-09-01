import cv2
from PIL import Image

from ultralytics import YOLO

model = YOLO("yolo11x.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True)  # Display preds. Accepts all YOLO predict arguments

# # from PIL
# im1 = Image.open("bus.jpg")
# results = model.predict(source=im1, save=True)  # save plotted images

# from ndarray
img_path = "/data_all/cjj_node/ultralytics/demo_images/catdog.png"
im2 = cv2.imread(img_path)
results = model.predict(source=im2, save=True, save_txt=False)  # save predictions as labels
