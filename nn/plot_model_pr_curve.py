import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import glob
import cv2
import torch
from ultralytics import YOLO
import os

# Set IoU threshold
IOU_THRESHOLD = 0.5

# Trained YOLOv8 model paths
model_paths = [
    "/media/robot/7846E2E046E29DDE/ultralytics/runs/detect/using_mspa_c2f_gd_modTaskAlignedAssigner/20250124final_result/weights/best.pt", # MGDT
    "/media/robot/7846E2E046E29DDE/ultralytics/runs/detect/origin_yolo_using_yolov8npt/train/weights/best.pt", # Baseline
    "/media/robot/7846E2E046E29DDE/ultralytics/runs/detect/using_mspa_c2f/train5/weights/best.pt", # M
    "/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/runs/detect/train2/weights/best.pt", # GD
    "/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/runs/detect/train4/weights/best.pt", # T
    "/media/robot/7846E2E046E29DDE/ultralytics/runs/detect/using_mspa_c2f_gd/train/weights/best.pt", # MGD
    "/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/runs/detect/train5/weights/best.pt", # MT
    "/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/runs/detect/train6/weights/best.pt", # GDT
]
models = [YOLO(model_path) for model_path in model_paths]

# Dataset path
dataset_path = "/media/robot/7846E2E046E29DDE/piglet_pic_from_net/valid/"  # Replace with your dataset root path
image_folder = os.path.join(dataset_path, "images")
label_folder = os.path.join(dataset_path, "labels")

# Load test images
test_images = glob.glob(os.path.join(image_folder, "*.jpg"))

# Convert YOLOv8 annotation format to pixel coordinates
def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return [x1, y1, x2, y2]

# Load Ground Truth (GT), YOLO format: class_id x_center y_center width height
def load_ground_truth(annotation_path, img_width, img_height):
    gt_boxes = []
    if not os.path.exists(annotation_path):
        return gt_boxes  # Return empty if annotation file does not exist
    with open(annotation_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            bbox = yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)
            gt_boxes.append((*bbox, class_id))
    return gt_boxes

# Compute IoU
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    xi1, yi1 = max(x1, x1_gt), max(y1, y1_gt)
    xi2, yi2 = min(x2, x2_gt), min(y2, y2_gt)
    
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0

# Store Precision and Recall of all models
all_precisions = []
all_recalls = []

# Iterate over each model for evaluation
for model in models:
    y_true = []
    y_scores = []

    for img_path in test_images:
        img = cv2.imread(img_path)
        img_name = os.path.basename(img_path).replace(".jpg", ".txt")
        annotation_path = os.path.join(label_folder, img_name)  # Annotation file path
        h, w, _ = img.shape

        # Load GT
        gt_boxes = load_ground_truth(annotation_path, w, h)
        
        # Model inference
        results = model(img)
        detections = results[0].boxes.data.cpu().numpy()  # Predicted boxes
        for det in detections:
            x_min, y_min, x_max, y_max, conf, class_id = det
            
            # IoU matching
            matched = False
            for gt_box in gt_boxes:
                x_min_gt, y_min_gt, x_max_gt, y_max_gt, gt_class_id = gt_box
                if class_id == gt_class_id and compute_iou((x_min, y_min, x_max, y_max), (x_min_gt, y_min_gt, x_max_gt, y_max_gt)) >= IOU_THRESHOLD:
                    matched = True
                    break
            
            y_true.append(1 if matched else 0)  # 1: correct detection, 0: false positive
            y_scores.append(conf)  # Confidence score
    
    # Compute Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    all_precisions.append(precision)
    all_recalls.append(recall)

# Compute average P-R curve across models
min_recall = max(map(lambda r: min(r), all_recalls))  # Align minimum recall range
max_recall = 1.0
recall_values = np.linspace(min_recall, max_recall, num=100)

# Interpolation to align recall
interpolated_precisions = []
for precision, recall in zip(all_precisions, all_recalls):
    interpolated_precision = np.interp(recall_values, recall, precision)
    interpolated_precisions.append(interpolated_precision)

mean_precision = np.mean(interpolated_precisions, axis=0)

# Compute mean AUC
average_auc = auc(recall_values, mean_precision)

# Labels for each model
labels = [
    "YOLOv8n + MSPA C2f + GD + THead",  # Model 1
    "YOLOv8n",                         # Model 2
    "YOLOv8n + MSPA C2f",              # Model 3
    "YOLOv8n + GD",                    # Model 4
    "YOLOv8n + THead",                 # Model 5
    "YOLOv8n + MSPA C2f + GD",         # Model 6
    "YOLOv8n + MSPA C2f + THead",      # Model 7
    "YOLOv8n + GD + THead"             # Model 8
]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# Plot P-R curves
for i, (precision, recall) in enumerate(zip(all_precisions, all_recalls)):
    if i < len(labels):
        if i == 0:
            # Highlight the first curve
            plt.plot(recall, precision, linestyle="-", color=colors[i], label=labels[i], alpha=1, linewidth=2)
        else:
            plt.plot(recall, precision, linestyle="-", color=colors[i], label=labels[i], alpha=1, linewidth=2)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve of each model based on YOLOv8n')
plt.legend()
plt.grid()
plt.show()
