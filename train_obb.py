from ultralytics import YOLO
 
# 加载预训练的模型
# model = YOLO("yolo11m-obb.yaml").load("weights/yolo11m-obb.pt")
model = YOLO('ultralytics/cfg/models/11/yolo11-obb.yaml')
model.load('my_install_model/yolo11x-obb.pt')  #加载预训练权重
 
# 定义训练参数，添加默认值、范围和中文注释
train_params = {
    'data': "data/bbu_ground_bbu/data.yaml",   
    "imgsz":1024,
	"epochs":500,
	"batch":8,
	"workers":8,  
	"device":4,   #没显卡则将0修改为'cpu'
	"optimizer":'SGD',
    "amp" : False,
	"cache":False,   #服务器可设置为True，训练速度变快
	"project" : 'runs/obb/train',
	"name" : "exp",
}
 
# 进行训练
results = model.train(**train_params)