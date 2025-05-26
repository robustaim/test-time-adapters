from ultralytics import YOLO
import wandb
import os

#wandb
wandb.login(key="5c8faed00414deece85b4f4fa1351764647eb2a9")
proj = "yolo_with_imagenetvid"
proj_name = "yolo11n_6"

# YOLO 모델
dataset_path = f"{os.getcwd()}/dataset/real_data.yaml"
model = YOLO("yolo11x.yaml")
batch_size = 16
devices = 'cuda:0'
optim = 'AdamW'
moment = 0.9 #momentum
lr_0 = 1e-3
lr_fit = 3e-2

result = model.train(data=dataset_path, epochs=1000, project=proj, name=proj_name,batch=batch_size, plots=True, resume=True, optimizer=optim,lr0=lr_0, lrf=lr_fit, momentum=moment,device = devices)

# python train.py | tee output.log