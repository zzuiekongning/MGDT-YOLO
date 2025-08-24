import torch
import torch.nn as nn

from ultralytics import YOLO

print("************************start")
# Initialize the YOLO model
model = YOLO('/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/ultralytics/models/v8/thead_yolov8.yaml')
print("************************")

# Load pre-trained weights
model.load('yolov8n.pt')

# Train the model
data_path = '/media/robot/7846E2E046E29DDE/piglet_pic_from_net/data.yaml'  # Path to the dataset configuration file
model.train(
    data=data_path,          # Dataset configuration path
    cfg = '/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/ultralytics/yolo/cfg/default.yaml'
)
