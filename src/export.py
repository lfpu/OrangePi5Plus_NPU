import numpy as np
import cv2
from rknn.api import RKNN
import time

modelPath = 'model.onnx'  # 假设模型路径为'rknn_model.rknn'

rknn = RKNN()
rknn.config(mean_values=[[0, 0, 0]], std_values=[[128, 128, 128]], target_platform='rk3588')
# 加载RKNN模型,inputs 为模型的输入名，以及设置输入的shape
ret = rknn.load_onnx(modelPath,inputs=['input.1'], input_size_list=[[1, 3, 112, 112]])
rknn.build(do_quantization=False)
print("build RKNN model successfully!")
# 构建RKNN模型
rknn.export_rknn('1.rknn')
print('RKNN model exported successfully!')

img=cv2.imread("./src/1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
img = cv2.resize(img, (112, 112))  # 假设模型输入尺寸为112x112
img = img.transpose(2, 0, 1)  # HWC -> CHW
img = np.expand_dims(img, 0)  # [3,112,112] -> [1,3,112,112]
img = img.astype(np.float32)  # 保证类型和模型一致

outputs = rknn.inference(inputs=[img])
