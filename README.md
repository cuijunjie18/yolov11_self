# Huawei工业质检-yolo处理方案

## 日志

- 2025-8-28
  - 添加cpri处理(4类别处理)

- 2025-8-29
  - 使用学长的6类别处理，训练cpri.
  - 发现之前的训练代码有bug,使用了默认的scale = n.

- 2025-9-9
  - 添加一些技巧，如获取标注的推理图、labels.

## 收获

### yolov11数据集格式

data.yaml文件
```py
train: /data_all/cjj_node/ultralytics/data/cpri2/images/train  # train images (relative to 'path') 128 images
val: /data_all/cjj_node/ultralytics/data/cpri2/images/val  # val images (relative to 'path') 128 images
test: /data_all/cjj_node/ultralytics/data/cpri2/images/test

nc: 6
 
# Classes
names: ['cpri', 'cpri_not_screw', 'cpri_screw', 'power', 'power_not_screw', 'power_screw']
```

目录格式
```txt
.
├── data.yaml
├── images
├── labels
├── raw_images
└── raw_labels

===其中images、labels如下
images
├── test
├── train
└── val

labels
├── test
├── train
└── val
```

### 标准训练代码

```py
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
	model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')
	model.load('my_install_model/yolo11x.pt')  #加载预训练权重
	model.train(data='data/cpri2/data.yaml',   #数据集yaml文件
	            imgsz=640,
	            epochs=500,
	            batch=8,
	            workers=8,  
	            device=1,   #没显卡则将0修改为'cpu'
	            optimizer='SGD',
                amp = False,
	            cache=False,   #服务器可设置为True，训练速度变快
				project = 'runs/cpri_od',
				name = "exp",
			)
```

输出
```shell
Overriding model.yaml nc=80 with nc=6
WARNING ⚠️ no model scale passed. Assuming scale='n'.
```

这是原来的代码，因为ultralytics/cfg/models/11目录下只有yolo11.yaml，这是通配文件，需要外部指明scale或代码指明，进行下面的修改

```py
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
	model = YOLO('ultralytics/cfg/models/11/yolo11x.yaml')   # 修改为yolo11x.yaml
	model.load('my_install_model/yolo11x.pt')  #加载预训练权重
	model.train(data='data/cpri2/data.yaml',   #数据集yaml文件
	            imgsz=640,
	            epochs=500,
	            batch=8,
	            workers=8,  
	            device=1,   #没显卡则将0修改为'cpu'
	            optimizer='SGD',
                amp = False,
	            cache=False,   #服务器可设置为True，训练速度变快
				project = 'runs/cpri_od',
				name = "exp",
			)
```

