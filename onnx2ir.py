from openvino.runtime import Core
from openvino.offline_transformations import serialize

ie = Core()
onnx_model_path = "models/exp2/resnet18_3.onnx"
model_onnx = ie.read_model(model=onnx_model_path)
compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")
# 使用 .serialize() 将 ONNX 模型导出到 IR
serialize(model=model_onnx, xml_path="models/exp2/resnet18_3.xml", bin_path="models/exp2/resnet18_3.bin",version = 'IR_V11')