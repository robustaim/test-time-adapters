from ultralytics import YOLO
import wandb
import os

# wandb
os.system("yolo settings wandb=True")
wandb.login()
proj = "yolo_pretraining"
model_name = "yolo11m"

# Hyperparameters
dataset_path = f"{os.getcwd()}/dataset/real_data.yaml"
batch_size = 30
devices = "cuda:0"
optim = "AdamW"
decay = 5e-4
lr = 1e-3

# YOLO
if not os.path.isdir(f"./{proj}"):
    model = YOLO(f"./pretrained/{model_name}.pt")
    resume = False
else:
    model = YOLO(f"./{proj}/{model_name}/weights/best.pt")
    resume = True

result = model.train(
    data=dataset_path, pretrained=True,
    epochs=100, batch=batch_size, imgsz=640, exist_ok=True,
    project=proj, name=model_name, plots=True, resume=resume,
    optimizer=optim, lr0=lr, cos_lr=True, weight_decay=decay,
    warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
    device=devices, fraction=1, save_period=1
)

print(f"Result: {result}")
