import torch
import torch.nn as nn

from ultralytics import YOLO

print("************************start")
model = YOLO('/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/ultralytics/models/v8/mspa_c2f_thead_yolov8.yaml')
print("************************")

model.load('yolov8n.pt')

# # Train the model
data_path = '/media/robot/7846E2E046E29DDE/piglet_pic_from_net/data.yaml'  # Replace with dataset config file path
model.train(
    data=data_path,          # Dataset configuration path
    cfg='/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/ultralytics/yolo/cfg/default.yaml'
)
