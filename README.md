# RKNN Model Inference Example

本项目演示如何在 OrangePi 5 Plus 上使用 RKNN Toolkit 对 ONNX 模型进行转换和推理。

## 目录结构

```
├── Dockerfile
├── src/
│   ├── model.onnx         # ONNX 格式的模型文件
│   ├── model.rknn         # RKNN 格式的模型文件（可选）
│   ├── rknnTest.py        # 主推理脚本
│   └── test.jpg           # 测试图片
```

## 依赖环境
- Python 3.x
- numpy
- opencv-python
- rknn-toolkit

建议在 OrangePi 5 Plus 或支持 RKNN 的设备上运行。

## 快速开始

1. 安装依赖：
   ```bash
   pip install numpy opencv-python rknn-toolkit
   ```

2. 运行推理脚本：
   ```bash
   python src/rknnTest.py
   ```

## rknnTest.py 说明

- 加载 ONNX 模型并转换为 RKNN 格式
- 初始化 RKNN 运行环境
- 读取并预处理测试图片
- 执行模型推理并输出结果 shape

## 注意事项
- 请确保 `model.onnx` 和 `test.jpg` 文件已放置在 `src/` 目录下。
- 若遇到模型或图片加载失败，请检查文件路径和格式。

## 参考
- [RKNN Toolkit 官方文档](https://github.com/rockchip-linux/rknn-toolkit)
