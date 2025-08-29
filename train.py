import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
	model = YOLO('ultralytics/cfg/models/11/yolo11x.yaml')   # 修改yaml
	model.load('my_install_model/yolo11x.pt')  #加载预训练权重
	model.train(data='data/cpri/data.yaml',   #数据集yaml文件
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