from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/media/robot/7846E2E046E29DDE/ultralytics/runs/detect/train5/weights/best.pt')  
    model.val(data='/media/robot/7846E2E046E29DDE/piglet_pic_from_net/data.yaml',
              )