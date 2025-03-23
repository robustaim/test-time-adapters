from ultralytics import YOLO
import wandb
import os

# wandb
wandb.login()
proj = "yolo_pretraining"
model_name = "yolo11m"

# Hyperparameters
dataset_path = f"{os.getcwd()}/dataset/real_data.yaml"
batch_size = 32
devices = 'cuda:0'
optim = 'AdamW'
moment = 0.9  # momentum
lr_0 = 1e-5
lr_fit = 1e-6

# YOLO
if not os.path.isdir(f"./{proj}/{model_name}"):
    pretrained = YOLO(f"./pretrained/{model_name}.pt")
    model = YOLO(f"{model_name}.yaml")

    for k, v in pretrained.state_dict().items():
        # 분류 레이어(detection head)를 제외한 가중치만 복사
        if 'head.cls' not in k and 'head.reg' not in k and k in model.state_dict():
            if model.state_dict()[k].shape == v.shape:
                model.state_dict()[k] = v.clone()
            else:
                print(f"Shape mismatch in layer {k}: {v.shape} vs {model.state_dict()[k].shape}, skipping")
else:
    model = YOLO(f"./{proj}/{model_name}/weights/best.pt")


if __name__ == '__main__':
    print(f"Loaded Model: {model.model}")

    result = model.train(
        data=dataset_path,
        epochs=10000, batch=batch_size, imgsz=640, fraction=0.005,
        project=proj, name=model_name, plots=True, resume=False,
        optimizer=optim, lr0=lr_0, lrf=lr_fit, momentum=moment,
        device=devices
    )

    print(f"Result: {result}")

# python train_from_pretrained.py | tee output.log
