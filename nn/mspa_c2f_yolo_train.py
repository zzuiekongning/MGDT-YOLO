import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.modules import (MSPA_C2f, SPRModule)

def initialize_weights(module):
    for sub_module in module.modules():  # 遍历 MSPA_C2f 的所有子模块
        """根据模块类型进行初始化"""
        if isinstance(module, nn.Conv2d):
            # 对卷积层使用Kaiming初始化（适用于ReLU激活）
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        elif isinstance(module, nn.BatchNorm2d):
            # BatchNorm层初始化
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        
        elif isinstance(module, nn.Linear):
            # 对全连接层使用Xavier初始化
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, SPRModule):
            # 对SPRModule进行初始化
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                # AdaptiveAvgPool2d 和 ReLU, Sigmoid等不需要初始化

print("************************")
# 创建模型实例
model = YOLO('/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/ultralytics/models/v8/mspa_c2f_yolov8.yaml')
#model = ModifiedYOLO(cfg='/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/ultralytics/models/v8/MGDT-yolov8.yaml')
print("************************")

model.load('/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/yolov8n.pt')
pytorch_model = model.model

for module in pytorch_model.modules():
    if isinstance(module, MSPA_C2f):  # 检查是否是自定义模块
        initialize_weights(module)

# # 训练模型
data_path = '/media/robot/7846E2E046E29DDE/piglet_pic_from_net/data.yaml'  # 替换为数据集配置文件路径
model.train(
    model='/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/yolov8n.pt',  # 改进模型的权重文件路径
    data=data_path,          # 数据集配置路径
    cfg = '/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/ultralytics/yolo/cfg/default.yaml'
)