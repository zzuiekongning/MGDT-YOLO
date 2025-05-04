import os
import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.metrics import r2_score

# 1. 初始化模型
model = YOLO("/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/runs/detect/train6/weights/best.pt")  # 替换为你的权重文件路径

# 2. 设置测试集文件夹路径
test_image_folder = "/media/robot/7846E2E046E29DDE/piglet_pic_from_net/valid/images"
test_label_folder = "/media/robot/7846E2E046E29DDE/piglet_pic_from_net/valid/labels"  # YOLO格式的标签文件夹（每张图片对应一个.txt文件）

# 3. 定义一个函数来将归一化坐标还原到原始图片大小
def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return [x1, y1, x2, y2]

# 4. 定义一个函数来计算 IoU（交并比）
def iou(box1, box2):
    # box格式: [x1, y1, x2, y2]
    # print(box1[0])
    # print(box2[0])
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# 5. 初始化存储列表
true_counts_class0 = []
true_counts_class1 = []
pred_counts_class0 = []
pred_counts_class1 = []

TP_class0 = FP_class0 = FN_class0 = 0
TP_class1 = FP_class1 = FN_class1 = 0

# 6. 遍历测试集，计算所有指标
image_files = [f for f in os.listdir(test_image_folder) if f.endswith(('.jpg'))]

for img_file in image_files:
    img_path = os.path.join(test_image_folder, img_file)
    label_path = os.path.join(test_label_folder, os.path.splitext(img_file)[0] + ".txt")

    # 读取图片的宽高
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]

    # 读取真实目标
    true_boxes_class0 = []
    true_boxes_class1 = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.split())
                bbox = yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)
                if class_id == 0:
                    true_boxes_class0.append(bbox)
                elif class_id == 1:
                    true_boxes_class1.append(bbox)

    true_counts_class0.append(len(true_boxes_class0))
    true_counts_class1.append(len(true_boxes_class1))

    # 使用模型预测
    results = model(img_path)
    detections = results[0].boxes

    pred_boxes_class0 = []
    pred_boxes_class1 = []
    for box in detections:
        class_id = int(box.cls.cpu().numpy())
        bbox = box.xyxy.cpu().numpy().tolist()  # 转为 [x1, y1, x2, y2]
        if class_id == 0:
            pred_boxes_class0.append(bbox)
        elif class_id == 1:
            pred_boxes_class1.append(bbox)

    pred_counts_class0.append(len(pred_boxes_class0))
    pred_counts_class1.append(len(pred_boxes_class1))

    # 计算 TP, FP, FN
    # 对类别 0
    matched_predictions = set()
    for gt_box in true_boxes_class0:
        found_match = False
        for i, pred_box in enumerate(pred_boxes_class0):
            # 解包嵌套列表，确保 pred_box 是 [x1, y1, x2, y2]
            pred_box = pred_box[0] if isinstance(pred_box[0], list) else pred_box
            if iou(gt_box, pred_box) > 0.5:
                TP_class0 += 1
                found_match = True
                matched_predictions.add(i)
                break
        if not found_match:
            FN_class0 += 1
    FP_class0 += len(pred_boxes_class0) - len(matched_predictions)


    # 对类别 1 
    matched_predictions = set()
    for gt_box in true_boxes_class1:
        found_match = False
        for i, pred_box in enumerate(pred_boxes_class1):
            # 解包嵌套列表，确保 pred_box 是 [x1, y1, x2, y2]
            pred_box = pred_box[0] if isinstance(pred_box[0], list) else pred_box
            if iou(gt_box, pred_box) > 0.5:
                TP_class1 += 1
                found_match = True
                matched_predictions.add(i)
                break
        if not found_match:
            FN_class1 += 1
    FP_class1 += len(pred_boxes_class1) - len(matched_predictions)


# 7. 计算 R²
r2_class0 = r2_score(true_counts_class0, pred_counts_class0) if len(true_counts_class0) > 1 else 0
print("***********************")
print(true_counts_class1)
print(pred_counts_class1)
print("***********************")
r2_class1 = r2_score(true_counts_class1, pred_counts_class1) if len(true_counts_class1) > 1 else 0

# 8. 计算平均准确率
# accuracy_class0 = TP_class0 / (TP_class0 + FP_class0 + FN_class0) if (TP_class0 + FP_class0 + FN_class0) > 0 else 0
# accuracy_class1 = TP_class1 / (TP_class1 + FP_class1 + FN_class1) if (TP_class1 + FP_class1 + FN_class1) > 0 else 0

# 9. 输出结果
print(f"Class 0 (目标类别0):")
print(f"  GT (真实目标数): {sum(true_counts_class0)}")
print(f"  TP: {TP_class0}")
print(f"  FP: {FP_class0}")
print(f"  FN: {FN_class0}")
print(f"  R²: {r2_class0:.2f}")
#print(f"  Mean Accuracy: {accuracy_class0:.2f}")

print(f"\nClass 1 (目标类别1):")
print(f"  GT (真实目标数): {sum(true_counts_class1)}")
print(f"  TP: {TP_class1}")
print(f"  FP: {FP_class1}")
print(f"  FN: {FN_class1}")
print(f"  R²: {r2_class1:.2f}")
#print(f"  Mean Accuracy: {accuracy_class1:.2f}")
