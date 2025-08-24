import torch
import torch.nn as nn

from ultralytics import YOLO

print("************************")
# Create model instance
model = YOLO('/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/ultralytics/models/v8/mspa_c2f_gd_yolov8.yaml')
# model = ModifiedYOLO(cfg='/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/ultralytics/models/v8/MGDT-yolov8.yaml')
print("************************")

model.load('/media/robot/7846E2E046E29DDE/ultralytics/runs/detect/using_mspa_c2f/train5/weights/best.pt')

# # Train the model
data_path = '/media/robot/7846E2E046E29DDE/piglet_pic_from_net/data.yaml'  # Replace with dataset config file path
model.train(
    # model='/media/robot/7846E2E046E29DDE/ultralytics/runs/detect/using_mspa_c2f/train5/weights/best.pt',  # Path to improved model weights
    data=data_path,          # Dataset configuration path
    cfg='/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/ultralytics/yolo/cfg/default.yaml'
)
