from ultralytics import YOLOv10
import os
current_directory = os.getcwd()
print("當前工作目錄:", current_directory)

# model = YOLOv10()
# # If you want to finetune the model with pretrained weights, you could load the 
# # pretrained weights like below
# # model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# # or
# # wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLOv10('/home/ldsc/chuweihuang/src/yolov10/yolov10n.pt')

# model.train(data='data.yaml', epochs=50, batch=4, imgsz=640, device = '0')
from ultralytics import YOLOv10
import os

# 使用絕對路徑
weights_path = '/home/ldsc/chuweihuang/src/yolov10/yolov10n.pt'

# 檢查文件是否存在
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"權重文件不存在：{weights_path}")

# 使用正確的文件路徑
model = YOLOv10(weights_path)

model.train(data='data.yaml', epochs=50, batch=4, imgsz=640, device='0')