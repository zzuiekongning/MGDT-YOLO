import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import glob
import cv2
import torch
from ultralytics import YOLO
import os

# 设定 IoU 阈值
IOU_THRESHOLD = 0.5

# 训练好的 YOLOv8 模型路径
model_paths = ["/media/robot/7846E2E046E29DDE/ultralytics/runs/detect/using_mspa_c2f_gd_modTaskAlignedAssigner/20250124final_result/weights/best.pt", #MGDT
"/media/robot/7846E2E046E29DDE/ultralytics/runs/detect/origin_yolo_using_yolov8npt/train/weights/best.pt",#Baseline
"/media/robot/7846E2E046E29DDE/ultralytics/runs/detect/using_mspa_c2f/train5/weights/best.pt",#M
"/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/runs/detect/train2/weights/best.pt", #GD
"/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/runs/detect/train4/weights/best.pt", #T
"/media/robot/7846E2E046E29DDE/ultralytics/runs/detect/using_mspa_c2f_gd/train/weights/best.pt",#MGD
"/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/runs/detect/train5/weights/best.pt",#MT
"/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/runs/detect/train6/weights/best.pt",#GDT

]  # 替换为你的模型
models = [YOLO(model_path) for model_path in model_paths]

# 设定数据集路径
dataset_path = "/media/robot/7846E2E046E29DDE/piglet_pic_from_net/valid/"  # 替换为你的数据集根目录
image_folder = os.path.join(dataset_path, "images")
label_folder = os.path.join(dataset_path, "labels")

# 读取测试集图片路径
test_images = glob.glob(os.path.join(image_folder, "*.jpg"))

# 归一化 YOLOv8 标注格式转换为像素坐标
def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return [x1, y1, x2, y2]

# 读取 Ground Truth（GT），YOLO 格式：class_id x_center y_center width height
def load_ground_truth(annotation_path, img_width, img_height):
    gt_boxes = []
    if not os.path.exists(annotation_path):
        return gt_boxes  # 如果没有标注文件，则返回空列表
    with open(annotation_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            bbox = yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)
            gt_boxes.append((*bbox, class_id))
    return gt_boxes

# 计算 IoU
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

# 存储所有模型的 Precision 和 Recall
all_precisions = []
all_recalls = []

# 遍历每个模型进行预测
for model in models:
    y_true = []
    y_scores = []

    for img_path in test_images:
        img = cv2.imread(img_path)
        img_name = os.path.basename(img_path).replace(".jpg", ".txt")
        annotation_path = os.path.join(label_folder, img_name)  # 获取对应的标注文件路径
        h, w, _ = img.shape

        # 读取 Ground Truth
        gt_boxes = load_ground_truth(annotation_path, w, h)
        
        # 进行预测
        results = model(img)
        detections = results[0].boxes.data.cpu().numpy()  # 获取预测框
        for det in detections:
            x_min, y_min, x_max, y_max, conf, class_id = det
            
            # 计算 IoU 进行匹配
            matched = False
            for gt_box in gt_boxes:
                x_min_gt, y_min_gt, x_max_gt, y_max_gt, gt_class_id = gt_box
                if class_id == gt_class_id and compute_iou((x_min, y_min, x_max, y_max), (x_min_gt, y_min_gt, x_max_gt, y_max_gt)) >= IOU_THRESHOLD:
                    matched = True
                    break
            
            y_true.append(1 if matched else 0)  # 1: 正确检测，0: 误检
            y_scores.append(conf)  # 置信度作为预测分数
    
    # 计算 Precision-Recall 曲线
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    all_precisions.append(precision)
    all_recalls.append(recall)

# 计算所有模型的平均 P-R 曲线
min_recall = max(map(lambda r: min(r), all_recalls))  # 统一最小 Recall 范围
max_recall = 1.0
recall_values = np.linspace(min_recall, max_recall, num=100)

# 进行插值，使不同模型的 Recall 统一
interpolated_precisions = []
for precision, recall in zip(all_precisions, all_recalls):
    interpolated_precision = np.interp(recall_values, recall, precision)
    interpolated_precisions.append(interpolated_precision)

mean_precision = np.mean(interpolated_precisions, axis=0)

# 计算平均 AUC
average_auc = auc(recall_values, mean_precision)

# 定义每条曲线的标签
labels = [
    "YOLOv8n + MSPA C2f + GD + THead",  # 第一条曲线
    "YOLOv8n",                         # 第二条曲线
    "YOLOv8n + MSPA C2f",              # 第三条曲线
    "YOLOv8n + GD",                    # 第四条曲线
    "YOLOv8n + THead",                 # 第五条曲线
    "YOLOv8n + MSPA C2f + GD",         # 第六条曲线
    "YOLOv8n + MSPA C2f + THead",      # 第七条曲线
    "YOLOv8n + GD + THead"             # 第八条曲线
]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


for i, (precision, recall) in enumerate(zip(all_precisions, all_recalls)):
    if i < len(labels):  # 确保索引不会越界
        if i == 0:
            # 第一条曲线使用蓝色，更加显眼
            plt.plot(recall, precision, linestyle="-", color=colors[i], label=labels[i], alpha=1, linewidth=2)
        else:
            # 其他曲线使用指定的颜色列表
            plt.plot(recall, precision, linestyle="-", color=colors[i], label=labels[i], alpha=1, linewidth=2)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve of each model based on YOLOv8n')
plt.legend()
plt.grid()
plt.show()



"""
# 绘制平均 P-R 曲线
colors = plt.get_cmap("tab10").colors

# 绘制 P-R 曲线
plt.figure(figsize=(10, 7))

for i, (precision, recall) in enumerate(zip(all_precisions, all_recalls)):
    if i == 1:
        # 突出显示第1条曲线（i == 1）
        plt.plot(recall, precision, linestyle="-", color=colors[i % len(colors)], label=f'Model {i+1}', alpha=1, linewidth=3)  # 加粗且为实线
    else:
        plt.plot(recall, precision, linestyle="-", color=colors[i % len(colors)], label=f'Model {i+1}', alpha=1, linewidth=3)

# 绘制平均 P-R 曲线（加粗）
#plt.plot(recall_values, mean_precision, color="black", linewidth=2, label=f'Average P-R (AUC={average_auc:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve of each model based on YOLOv8n')
plt.legend()
plt.grid()
plt.show()
"""



'''
for i, (precision, recall) in enumerate(zip(all_precisions, all_recalls)):
    plt.plot(recall, precision, linestyle="--", color=colors[i % len(colors)], label=f'Model {i+1}', alpha=0.7)

# 绘制平均 P-R 曲线（加粗）
#plt.plot(recall_values, mean_precision, color="black", linewidth=2, label=f'Average P-R (AUC={average_auc:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve of each model based on YOLOv8n')
plt.legend()
plt.grid()
plt.show()
'''

