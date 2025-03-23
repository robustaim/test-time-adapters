from ultralytics import YOLO
import wandb
import os

# wandb
wandb.login()
proj = "yolo_pretraining"
model_name = "yolo11m"

# Hyperparameters
dataset_path = f"{os.getcwd()}/apt/dataset/real_data.yaml"
batch_size = 32
devices = 'cuda:0'
optim = 'AdamW'
moment = 0.9  # momentum
lr_0 = 1e-5
lr_fit = 1e-6

# YOLO
model = YOLO(f"{model_name}.yaml")

result = model.train(
    data=dataset_path,
    epochs=1000, batch=batch_size, imgsz=640, fraction=0.005,
    project=proj, name=model_name, plots=True, resume=True,
    optimizer=optim, lr0=lr_0, lrf=lr_fit, momentum=moment,
    device=devices
)

print(f"Result: {result}")

# python train.py | tee output.log
