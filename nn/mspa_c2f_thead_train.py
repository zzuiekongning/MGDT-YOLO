import torch
import torch.nn as nn

from ultralytics import YOLO

print("************************start")
model = YOLO('/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/ultralytics/models/v8/mspa_c2f_thead_yolov8.yaml')
print("************************")

model.load('yolov8n.pt')

# # 训练模型
data_path = '/media/robot/7846E2E046E29DDE/piglet_pic_from_net/data.yaml'  # 替换为数据集配置文件路径
model.train(
    data=data_path,          # 数据集配置路径
    cfg = '/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/ultralytics/yolo/cfg/default.yaml'
)