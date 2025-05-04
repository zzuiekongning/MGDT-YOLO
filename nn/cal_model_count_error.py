import os
import numpy as np
from ultralytics import YOLO

# EPS = 1e-12
# 1. 初始化模型
model = YOLO("/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/runs/detect/train6/weights/best.pt")  # 替换为你的权重文件路径

# 2. 设置测试集文件夹路径
test_image_folder = "/media/robot/7846E2E046E29DDE/piglet_pic_from_net/valid/images"
test_label_folder = "/media/robot/7846E2E046E29DDE/piglet_pic_from_net/valid/labels" 

# 3. 遍历测试集，获取真实计数和预测计数
image_files = [f for f in os.listdir(test_image_folder) if f.endswith(('.jpg'))]
true_counts_class0 = []
true_counts_class1 = []
pred_counts_class0 = []
pred_counts_class1 = []

for img_file in image_files:
    # 图片路径和对应的标注文件路径
    img_path = os.path.join(test_image_folder, img_file)
    label_path = os.path.join(test_label_folder, os.path.splitext(img_file)[0] + ".txt")

    # 读取真实目标数
    count_class0 = 0
    count_class1 = 0
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                class_id = int(line.split()[0])  # YOLO格式的类别ID在每行的第一个字段
                if class_id == 0:
                    count_class0 += 1
                elif class_id == 1:
                    count_class1 += 1
    true_counts_class0.append(count_class0)
    true_counts_class1.append(count_class1)

    # 使用模型推理并统计预测目标数
    results = model(img_path)
    detections = results[0].boxes

    count_pred_class0 = sum(1 for box in detections if box.cls == 0)
    count_pred_class1 = sum(1 for box in detections if box.cls == 1)
    pred_counts_class0.append(count_pred_class0)
    pred_counts_class1.append(count_pred_class1)
print(true_counts_class0)
print (pred_counts_class0)

# 计算误差函数（处理真实计数为0的情况）
def calculate_errors(true_counts, pred_counts):
    true_counts = np.array(true_counts)
    pred_counts = np.array(pred_counts)
    ae = np.abs(pred_counts - true_counts)  # 绝对误差
    mae = np.mean(ae)  # 平均绝对误差
    mse = np.mean(ae ** 2)  # 均方误差
    
    # 对于真实计数为0的情况，跳过该数据点
    non_zero_mask = true_counts > 0
    if np.any(non_zero_mask):  # 确保有非零的真实计数
        mape = np.mean(ae[non_zero_mask] / true_counts[non_zero_mask]) * 100  # 只计算非零部分
    else:
        mape = 0  # 如果所有真实计数都为0，MAPE默认为0
    
    return mae, mse, mape

# 按类别计算误差（调用修改后的函数）
mae0, mse0, mape0 = calculate_errors(true_counts_class0, pred_counts_class0)
mae1, mse1, mape1 = calculate_errors(true_counts_class1, pred_counts_class1)

# 输出结果
print("Class 0 (目标类别0) 计数误差:")
print(f"  Mean Absolute Error (MAE): {mae0:.2f}")
print(f"  Mean Squared Error (MSE): {mse0:.2f}")
print(f"  Mean Absolute Percentage Error (MAPE): {mape0:.2f}%")

print("\nClass 1 (目标类别1) 计数误差:")
print(f"  Mean Absolute Error (MAE): {mae1:.2f}")
print(f"  Mean Squared Error (MSE): {mse1:.2f}")
print(f"  Mean Absolute Percentage Error (MAPE): {mape1:.2f}%")

# # 4. 计算计数误差
# def calculate_errors(true_counts, pred_counts):
#     ae = np.abs(np.array(pred_counts) - np.array(true_counts))  # 绝对误差
#     mae = np.mean(ae)  # 平均绝对误差
#     mse = np.mean(ae ** 2)  # 均方误差
#     mape = np.mean(ae / np.array(true_counts)) * 100  # 平均误差率
#     return mae, mse, mape

# # 按类别计算误差
# new_true_counts_class0 = [x+EPS for x in true_counts_class0]

# mae0, mse0, mape0 = calculate_errors(new_true_counts_class0, pred_counts_class0)
# mae1, mse1, mape1 = calculate_errors(true_counts_class1, pred_counts_class1)

# # 5. 输出结果
# print("Class 0 (目标类别0) 计数误差:")
# print(f"  Mean Absolute Error (MAE): {mae0:.2f}")
# print(f"  Mean Squared Error (MSE): {mse0:.2f}")
# print(f"  Mean Absolute Percentage Error (MAPE): {mape0:.2f}%")

# print("\nClass 1 (目标类别1) 计数误差:")
# print(f"  Mean Absolute Error (MAE): {mae1:.2f}")
# print(f"  Mean Squared Error (MSE): {mse1:.2f}")
# print(f"  Mean Absolute Percentage Error (MAPE): {mape1:.2f}%")

# 6. 可选：保存误差分析结果
# output_file = "counting_error_analysis.txt"
# with open(output_file, "w") as f:
#     f.write("Class 0 (目标类别0) 计数误差:\n")
#     f.write(f"  Mean Absolute Error (MAE): {mae0:.2f}\n")
#     f.write(f"  Mean Squared Error (MSE): {mse0:.2f}\n")
#     f.write(f"  Mean Absolute Percentage Error (MAPE): {mape0:.2f}%\n\n")
#     f.write("Class 1 (目标类别1) 计数误差:\n")
#     f.write(f"  Mean Absolute Error (MAE): {mae1:.2f}\n")
#     f.write(f"  Mean Squared Error (MSE): {mse1:.2f}\n")
#     f.write(f"  Mean Absolute Percentage Error (MAPE): {mape1:.2f}%\n")

# print(f"\n误差分析结果已保存到 {output_file}")
