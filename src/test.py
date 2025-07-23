import numpy as np
import cv2
from rknn.api import RKNN
import time

modelPath = '1.rknn'  # 假设模型路径为'rknn_model.rknn'

def read_img(image_path,size=112):
    image = cv2.imread(image_path)
    if image is None:
        print('Failed to read image!')
        exit(1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    image = cv2.resize(image, (size,size))  # 模型输入尺寸
    image= image.transpose(2, 0, 1)  # HWC -> CHW
    image = np.expand_dims(image, 0)  # [3,112,112] -> [1,3,112,112]
    image = image.astype(np.float32)  # 保证类型和模型一致
    return image
def similarity(outputs1, outputs2):
    # 计算余弦相似度
    norm1 = np.linalg.norm(outputs1)
    norm2 = np.linalg.norm(outputs2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_sim = np.dot(outputs1.flatten(), outputs2.flatten()) / (norm1 * norm2)
    return cos_sim


rknn = RKNN()
rknn.load_rknn(modelPath)

print('RKNN model loaded successfully!')
# 配置输入输出
ret = rknn.init_runtime(target="rk3588")
if ret != 0:
    print('Init runtime failed!')
    exit(ret)
# 读取图片
image1= read_img('test.jpg',size=112)
image2= read_img('test2.jpg',size=112)
# 输入尺寸为112x112
outputs1 = rknn.inference(inputs=[image1])
outputs2 = rknn.inference(inputs=[image2])
if outputs1 is None or outputs2 is None:
    print('Inference failed!')
    exit(1)

sim = similarity(outputs1[0], outputs2[0])
print(f'Similarity: {sim:.4f}')  # 输出结果的形状
rknn.release()
