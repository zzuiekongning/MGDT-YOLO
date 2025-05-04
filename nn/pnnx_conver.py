import torch
from ultralytics import YOLO
import pnnx

# 加载 PyTorch 训练好的模型
x= torch.rand(1,3,640,640)
model = YOLO("/media/robot/7846E2E046E29DDE/ultralytics/runs/detect/using_mspa_c2f_gd/train/weights/best.pt")

success = model.export(format="torchscript", simplify=True)  # export the model to onnx format

assert success