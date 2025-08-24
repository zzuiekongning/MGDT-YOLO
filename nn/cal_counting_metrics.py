import os
import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.metrics import r2_score

# 1. Initialize model
model = YOLO("/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/runs/detect/train6/weights/best.pt")  # Replace with your weight file path

# 2. Set test dataset folder paths
test_image_folder = "/media/robot/7846E2E046E29DDE/piglet_pic_from_net/valid/images"
test_label_folder = "/media/robot/7846E2E046E29DDE/piglet_pic_from_net/valid/labels"  # YOLO format labels folder (one .txt file per image)

# 3. Convert normalized YOLO coordinates to original image size
def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return [x1, y1, x2, y2]

# 4. Compute IoU (Intersection over Union)
def iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# 5. Initialize storage lists
true_counts_class0 = []
true_counts_class1 = []
pred_counts_class0 = []
pred_counts_class1 = []

TP_class0 = FP_class0 = FN_class0 = 0
TP_class1 = FP_class1 = FN_class1 = 0

# 6. Iterate over test dataset and compute metrics
image_files = [f for f in os.listdir(test_image_folder) if f.endswith(('.jpg'))]

for img_file in image_files:
    img_path = os.path.join(test_image_folder, img_file)
    label_path = os.path.join(test_label_folder, os.path.splitext(img_file)[0] + ".txt")

    # Read image size
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]

    # Read ground truth annotations
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

    # Model inference
    results = model(img_path)
    detections = results[0].boxes

    pred_boxes_class0 = []
    pred_boxes_class1 = []
    for box in detections:
        class_id = int(box.cls.cpu().numpy())
        bbox = box.xyxy.cpu().numpy().tolist()  # Convert to [x1, y1, x2, y2]
        if class_id == 0:
            pred_boxes_class0.append(bbox)
        elif class_id == 1:
            pred_boxes_class1.append(bbox)

    pred_counts_class0.append(len(pred_boxes_class0))
    pred_counts_class1.append(len(pred_boxes_class1))

    # Compute TP, FP, FN
    # For class 0
    matched_predictions = set()
    for gt_box in true_boxes_class0:
        found_match = False
        for i, pred_box in enumerate(pred_boxes_class0):
            # Unpack nested list if needed, ensure pred_box is [x1, y1, x2, y2]
            pred_box = pred_box[0] if isinstance(pred_box[0], list) else pred_box
            if iou(gt_box, pred_box) > 0.5:
                TP_class0 += 1
                found_match = True
                matched_predictions.add(i)
                break
        if not found_match:
            FN_class0 += 1
    FP_class0 += len(pred_boxes_class0) - len(matched_predictions)

    # For class 1
    matched_predictions = set()
    for gt_box in true_boxes_class1:
        found_match = False
        for i, pred_box in enumerate(pred_boxes_class1):
            # Unpack nested list if needed, ensure pred_box is [x1, y1, x2, y2]
            pred_box = pred_box[0] if isinstance(pred_box[0], list) else pred_box
            if iou(gt_box, pred_box) > 0.5:
                TP_class1 += 1
                found_match = True
                matched_predictions.add(i)
                break
        if not found_match:
            FN_class1 += 1
    FP_class1 += len(pred_boxes_class1) - len(matched_predictions)


# 7. Compute R² score
r2_class0 = r2_score(true_counts_class0, pred_counts_class0) if len(true_counts_class0) > 1 else 0
print("***********************")
print(true_counts_class1)
print(pred_counts_class1)
print("***********************")
r2_class1 = r2_score(true_counts_class1, pred_counts_class1) if len(true_counts_class1) > 1 else 0

# 8. Compute mean accuracy (optional, commented out)
# accuracy_class0 = TP_class0 / (TP_class0 + FP_class0 + FN_class0) if (TP_class0 + FP_class0 + FN_class0) > 0 else 0
# accuracy_class1 = TP_class1 / (TP_class1 + FP_class1 + FN_class1) if (TP_class1 + FP_class1 + FN_class1) > 0 else 0

# 9. Print results
print(f"Class 0:")
print(f"  GT (ground truth count): {sum(true_counts_class0)}")
print(f"  TP: {TP_class0}")
print(f"  FP: {FP_class0}")
print(f"  FN: {FN_class0}")
print(f"  R²: {r2_class0:.2f}")
# print(f"  Mean Accuracy: {accuracy_class0:.2f}")

print(f"\nClass 1:")
print(f"  GT (ground truth count): {sum(true_counts_class1)}")
print(f"  TP: {TP_class1}")
print(f"  FP: {FP_class1}")
print(f"  FN: {FN_class1}")
print(f"  R²: {r2_class1:.2f}")
# print(f"  Mean Accuracy: {accuracy_class1:.2f}")
`
##
