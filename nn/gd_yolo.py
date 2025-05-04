import torch
import torch.nn as nn

from ultralytics import YOLO

print("************************")
# 创建模型实例
model = YOLO('/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/ultralytics/models/v8/gd_yolov8.yaml')
#model = ModifiedYOLO(cfg='/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/ultralytics/models/v8/MGDT-yolov8.yaml')
print("************************")

model.load('yolov8n.pt')

# # 训练模型
data_path = '/media/robot/7846E2E046E29DDE/piglet_pic_from_net/data.yaml'  # 替换为数据集配置文件路径
model.train(
    # model='/media/robot/7846E2E046E29DDE/ultralytics/runs/detect/using_mspa_c2f/train5/weights/best.pt',  # 改进模型的权重文件路径
    data=data_path,          # 数据集配置路径
    cfg = '/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/ultralytics/yolo/cfg/default.yaml'
)