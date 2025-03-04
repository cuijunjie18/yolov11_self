import numpy as np
import cv2

img_path = "data/armors_data/unused_imgs/val/val/nred5407.jpg"
img = cv2.imread(img_path)
shape = img.shape
print(shape)
print(shape[::-1])