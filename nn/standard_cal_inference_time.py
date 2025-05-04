import cv2
import time
import torch
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
import os

# 1. 初始化模型
# 创建两个CUDA事件
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# 加载模型
model = YOLO("/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics/runs/detect/train6/weights/best.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型移动到GPU
model.to(device)

# 定义输入图像文件夹路径
input_folder = "/media/robot/7846E2E046E29DDE/piglet_pic_from_net/valid/images/"

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # 修改为模型要求的尺寸
    transforms.ToTensor(),
])

# 遍历文件夹中的所有图片
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg'))]

# 初始化推理时间列表
inference_times = []

for image_file in image_files:
    input_path = os.path.join(input_folder, image_file)
    
    # 加载并处理输入图像
    image = Image.open(input_path)  # 打开图片
    image_tensor = transform(image).unsqueeze(0).to(device) 

    # 开始记录时间
    start_event.record()

    # 执行推理过程
    with torch.no_grad():
        output = model(image_tensor)  # 使用 predict 方法进行推理

    # 结束记录时间
    end_event.record()
    torch.cuda.synchronize()

    # 计算时间差
    inference_time = start_event.elapsed_time(end_event)
    inference_times.append(inference_time)
    
    # 打印每张图片的推理时间
    print(f"Image: {image_file}, Inference Time: {inference_time:.2f} ms")

# 去掉最高和最低值后计算平均推理时间
if len(inference_times) > 2:
    inference_times.sort()
    filtered_times = inference_times[1:-1]  # 去掉最高和最低值
    average_inference_time = sum(filtered_times) / len(filtered_times)
    print(f"Average Inference Time (excluding max and min): {average_inference_time:.2f} ms")
elif len(inference_times) > 0:
    average_inference_time = sum(inference_times) / len(inference_times)
    print(f"Average Inference Time: {average_inference_time:.2f} ms")
else:
    print("No images found in the folder.")


# import cv2
# import time
# import torch
# from ultralytics import YOLO
# from PIL import Image
# from torchvision import transforms
# import os

# # 1. 初始化模型
# # 创建两个CUDA事件
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)

# # 加载模型
# model = YOLO("/media/robot/7846E2E046E29DDE/ultralytics/runs/detect/using_mspa_c2f/train5/weights/best.pt")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 将模型移动到GPU
# model.to(device)

# # 定义输入图像文件夹路径
# input_folder = "/media/robot/7846E2E046E29DDE/piglet_pic_from_net/valid/images/"

# # 定义图像预处理
# transform = transforms.Compose([
#     transforms.Resize((640, 640)),  # 修改为模型要求的尺寸
#     transforms.ToTensor(),
# ])

# # 遍历文件夹中的所有图片
# image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# # 初始化总时间和计数器
# total_inference_time = 0.0
# num_images = 0

# for image_file in image_files:
#     input_path = os.path.join(input_folder, image_file)
    
#     # 加载并处理输入图像
#     image = Image.open(input_path)  # 打开图片
#     image_tensor = transform(image).unsqueeze(0).to(device) 

#     # 开始记录时间
#     start_event.record()

#     # 执行推理过程
#     with torch.no_grad():
#         output = model(image_tensor)  # 使用 predict 方法进行推理

#     # 结束记录时间
#     end_event.record()
#     torch.cuda.synchronize()

#     # 计算时间差
#     inference_time = start_event.elapsed_time(end_event)
#     total_inference_time += inference_time
#     num_images += 1
    
#     # 打印每张图片的推理时间
#     print(f"Image: {image_file}, Inference Time: {inference_time:.2f} ms")

# # 计算平均推理时间
# if num_images > 0:
#     average_inference_time = total_inference_time / num_images
#     print(f"Average Inference Time: {average_inference_time:.2f} ms")
# else:
#     print("No images found in the folder.")



# import cv2
# import time
# import torch
# from ultralytics import YOLO
# from PIL import Image
# from torchvision import transforms

# # 1. 初始化模型
# # 创建两个CUDA事件
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)

# # 加载模型
# model = YOLO("/media/robot/7846E2E046E29DDE/ultralytics/runs/detect/using_mspa_c2f/train5/weights/best.pt")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 将模型移动到GPU
# model.to(device)

# # 加载并处理输入图像
# input_path = "/media/robot/7846E2E046E29DDE/piglet_pic_from_net/valid/images/0398.jpg"
# image = Image.open(input_path)  # 打开图片
# transform = transforms.Compose([
#     transforms.Resize((640, 640)),  # 修改为模型要求的尺寸
#     transforms.ToTensor(),
# ])
# image_tensor = transform(image).unsqueeze(0).to(device) 


# # 开始记录时间
# start_event.record()

# # 执行推理过程
# with torch.no_grad():
#     output = model(image_tensor)  # 使用 predict 方法进行推理

# # 结束记录时间
# end_event.record()
# torch.cuda.synchronize()

# # 计算时间差
# inference_time = start_event.elapsed_time(end_event)