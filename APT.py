import marimo

__generated_with = "0.11.2"
app = marimo.App(width="full", auto_download=["ipynb"])


@app.cell
def _(mo):
    mo.md("""# Pluggable TTA Implementation using light-weight sparse autoencoder""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Import Libraries""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from os import path, rename, mkdir, listdir, system

    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader

    from torchvision import datasets, utils
    from torchvision import transforms
    from ultralytics import YOLO

    import numpy as np
    import pandas as pd

    from tqdm.notebook import tqdm
    import matplotlib.pyplot as plt
    import pygwalker as pyg
    import wandb

    datasets.utils.tqdm = tqdm
    return (
        DataLoader,
        YOLO,
        datasets,
        listdir,
        mkdir,
        nn,
        np,
        optim,
        path,
        pd,
        plt,
        pyg,
        rename,
        system,
        torch,
        tqdm,
        transforms,
        utils,
        wandb,
    )


@app.cell
def _(wandb):
    # WandB Initialization
    wandb.init(project="plugin-TTA-ideaA")
    return


@app.cell
def _(mo):
    mo.md("""### Check GPU Availability""")
    return


@app.cell
def _(system):
    system("nvidia-smi")
    return


@app.cell
def _(torch):
    # Set CUDA Device Number 0~7
    DEVICE_NUM = 4
    ADDITIONAL_GPU = 1

    if torch.cuda.is_available():
        if ADDITIONAL_GPU:
            torch.cuda.set_device(DEVICE_NUM)
            device = torch.device("cuda")
        else:
            device = torch.device(f"cuda:{DEVICE_NUM}")
    else:
        device = torch.device("cpu")
        DEVICE_NUM = -1

    print(f"INFO: Using device - {device}" + (f":{DEVICE_NUM}" if ADDITIONAL_GPU else ""))
    return ADDITIONAL_GPU, DEVICE_NUM, device


@app.cell
def _(mo):
    mo.md("""## Load Dataset""")
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### GOT-10k Dataset for Next-frame Prediction Task (Default Pretraining Process)
        http://got-10k.aitestunion.com/downloads

        #### Data File Structure
        The downloaded and extracted full dataset should follow the file structure:
        ```
          |-- GOT-10k/
             |-- train/
             |  |-- GOT-10k_Train_000001/
             |  |   ......
             |  |-- GOT-10k_Train_009335/
             |  |-- list.txt
             |-- val/
             |  |-- GOT-10k_Val_000001/
             |  |   ......
             |  |-- GOT-10k_Val_000180/
             |  |-- list.txt
             |-- test/
             |  |-- GOT-10k_Test_000001/
             |  |   ......
             |  |-- GOT-10k_Test_000180/
             |  |-- list.txt
        ```

        #### Annotation Description
        Each sequence folder contains 4 annotation files and 1 meta file. A brief description of these files follows (let N denotes sequence length):

        * groundtruth.txt -- An N×4 matrix with each line representing object location [xmin, ymin, width, height] in one frame.
        * cover.label -- An N×1 array representing object visible ratios, with levels ranging from 0~8.
        * absense.label -- An binary N×1 array indicating whether an object is absent or present in each frame.
        * cut_by_image.label -- An binary N×1 array indicating whether an object is cut by image in each frame.
        * meta_info.ini -- Meta information about the sequence, including object and motion classes, video URL and more.
        * Values 0~8 in file cover.label correspond to ranges of object visible ratios: 0%, (0%, 15%], (15%~30%], (30%, 45%], (45%, 60%], (60%, 75%], (75%, 90%], (90%, 100%) and 100% respectively.
        """
    )
    return


@app.cell
def _(datasets, path, pd, tqdm):
    from typing import Callable, Optional

    datasets.utils.tqdm = tqdm


    class FoodImageDataset(datasets.ImageFolder):
        download_method = datasets.utils.download_and_extract_archive
        download_url = "https://www.kaggle.com/api/v1/datasets/download/trolukovich/food11-image-dataset"

        def __init__(self, root: str, force_download: bool = True, train: bool = True, valid: bool = False, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
            self.download(root, force=force_download)

            if train:
                if valid:
                    root = path.join(root, "validation")
                else:
                    root = path.join(root, "training")
            else:
                root = path.join(root, "evaluation")

            super().__init__(root=root, transform=transform, target_transform=target_transform)

        @classmethod
        def download(cls, root: str, force: bool = False):
            if force or not path.isfile(path.join(root, "archive.zip")):
                cls.download_method(cls.download_url, download_root=root, extract_root=root, filename="archive.zip")
                print("INFO: Dataset archive downloaded and extracted.")
            else:
                print("INFO: Dataset archive found in the root directory. Skipping download.")

        @property
        def df(self) -> pd.DataFrame:
            return pd.DataFrame(dict(path=[d[0] for d in self.samples], label=[self.classes[lb] for lb in self.targets]))
    return Callable, FoodImageDataset, Optional


@app.cell
def _(FoodImageDataset, augmenter, path, resizer):
    DATA_ROOT = path.join(".", "data", "food11")

    train_dataset = FoodImageDataset(root=DATA_ROOT, force_download=False, train=True, transform=augmenter)
    valid_dataset = FoodImageDataset(root=DATA_ROOT, force_download=False, valid=True, transform=resizer)
    test_dataset = FoodImageDataset(root=DATA_ROOT, force_download=False, train=False, transform=resizer)

    print(f"INFO: Dataset loaded successfully. Number of samples - Train({len(train_dataset)}), Valid({len(valid_dataset)}), Test({len(test_dataset)})")
    return DATA_ROOT, test_dataset, train_dataset, valid_dataset


@app.cell
def _(pyg, train_dataset):
    # Train Dataset Distribution
    pyg.walk(train_dataset.df)
    return


@app.cell
def _(pyg, train_dataset):
    # Valid Dataset Distribution
    pyg.walk(train_dataset.df)
    return


@app.cell
def _(mo):
    mo.md(r"""## DataLoader""")
    return


@app.cell
def _():
    # Set Batch Size
    BATCH_SIZE = 64, 64, 10
    return (BATCH_SIZE,)


@app.cell
def _(BATCH_SIZE, DataLoader, test_dataset, train_dataset, valid_dataset):
    MULTI_PROCESSING = True  # Set False if DataLoader is causing issues

    from platform import system
    if MULTI_PROCESSING and system() != "Windows":  # Multiprocess data loading is not supported on Windows
        import multiprocessing
        cpu_cores = multiprocessing.cpu_count()
        print(f"INFO: Number of CPU cores - {cpu_cores}")
    else:
        cpu_cores = 0
        print("INFO: Using DataLoader without multi-processing.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE[0], shuffle=True, num_workers=cpu_cores)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE[1], shuffle=False, num_workers=cpu_cores)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE[2], shuffle=False, num_workers=cpu_cores)
    return (
        MULTI_PROCESSING,
        cpu_cores,
        multiprocessing,
        system,
        test_loader,
        train_loader,
        valid_loader,
    )


@app.cell
def _(IMG_NORM, np, plt):
    # Image Visualizer
    def imshow(image_list, mean=IMG_NORM['mean'], std=IMG_NORM['std']):
        np_image = np.array(image_list).transpose((1, 2, 0))
        de_norm_image = np_image * std + mean
        plt.figure(figsize=(10, 10))
        plt.imshow(de_norm_image)
    return (imshow,)


@app.cell
def _(imshow, train_loader, utils):
    images, targets = next(iter(train_loader))
    grid_images = utils.make_grid(images, nrow=8, padding=10)
    imshow(grid_images)
    return grid_images, images, targets


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Define Model
        ### APT: Adaptive Plugin for TTA (Test-time Adaptation)
        """
    )
    return


@app.cell
def _(ViTConfig, ViTModel, nn):
    class ViTImageClassifier(nn.Module):
        def __init__(self, config: ViTConfig, num_classes: int):
            super().__init__()
            self.config = config
            self.vit = ViTModel(config=config)
            self.fc = nn.Linear(self.config.hidden_size, num_classes)

        def forward(self, x):
            out = self.vit(x)
            pooled = out.pooler_output  # [batch_size, hidden_size]
            logits = self.fc(pooled)  # [batch_size, num_classes]
            return logits
    return (ViTImageClassifier,)


@app.cell
def _(ADDITIONAL_GPU, DEVICE_NUM, ViTImageClassifier, device, nn):
    # Initialize Model
    model = ViTImageClassifier()

    if ADDITIONAL_GPU:
        model = nn.DataParallel(model, device_ids=list(range(DEVICE_NUM, DEVICE_NUM + ADDITIONAL_GPU + 1)))
        model
    else:
        model.to(device)
    return (model,)


@app.cell
def _():
    # Training
    return


@app.cell
def _(plt):
    from IPython.display import display
    import ipywidgets as widgets

    # Interactive Loss Plot Update
    def create_plot():
        train_losses, valid_losses = [], []

        # Enable Interactive Mode
        plt.ion()

        # Loss Plot Setting
        fig, ax = plt.subplots(figsize=(6, 2))
        train_line, = ax.plot(train_losses, label="Train Loss", color="purple")
        valid_line, = ax.plot(valid_losses, label="Valid Loss", color="red")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Cross Entropy Loss")

        # Display Plot
        plot = widgets.Output()
        display(plot)

        def update_plot(train_loss=None, valid_loss=None):
            if train_loss is not None:
                train_losses.append(train_loss)
            if valid_loss is not None:
                valid_losses.append(valid_loss)
            train_line.set_ydata(train_losses)
            train_line.set_xdata(range(len(train_losses)))
            valid_line.set_ydata(valid_losses)
            valid_line.set_xdata(range(len(valid_losses)))
            ax.relim()
            ax.autoscale_view()
            with plot:
                plot.clear_output(wait=True)
                display(fig)

        return update_plot
    return create_plot, display, widgets


@app.cell
def _():
    def avg(lst):
        try:
            return sum(lst) / len(lst)
        except ZeroDivisionError:
            return 0
    return (avg,)


@app.cell
def _(mo):
    mo.md(r"""## Default Pre-training Process""")
    return


@app.cell
def _(model, nn, optim, wandb):
    # Set Epoch Count & Learning Rate
    EPOCHS = 50
    LEARNING_RATE = 1e-4, 1e-6
    WEIGHT_DECAY = 0.05
    MAX_GRAD_NORM = 1.0
    USE_CACHE = False

    criterion = nn.CrossEntropyLoss()
    wandb.watch(model, criterion, log="all", log_freq=10)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE[0], weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE[1])
    return (
        EPOCHS,
        LEARNING_RATE,
        MAX_GRAD_NORM,
        USE_CACHE,
        WEIGHT_DECAY,
        criterion,
        optimizer,
        scheduler,
    )


@app.cell
def _(
    EPOCHS,
    USE_NORMAL,
    create_plot,
    criterion,
    device,
    model,
    optimizer,
    scheduler,
    torch,
    tqdm,
    train_dataset,
    train_loader,
    valid_loader,
    wandb,
):
    train_length, valid_length = map(len, (train_loader, valid_loader))
    if not USE_NORMAL:
        raise ValueError("Model is not set to Normal Vi-T. Please set USE_NORMAL to True.")

    epochs = tqdm(range(EPOCHS), desc="Running Epochs")
    with (tqdm(total=train_length, desc="Training") as train_progress,
            tqdm(total=valid_length, desc="Validation") as valid_progress):  # Set up Progress Bars
        update = create_plot()  # Create Loss Plot

        for epoch in epochs:
            train_progress.reset(total=train_length)
            valid_progress.reset(total=valid_length)

            train_acc, train_loss = 0, 0

            # Training
            model.train()
            for i, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item() / train_length
                corrects = (torch.max(outputs, 1)[1] == targets.data).sum()
                train_acc += corrects / len(train_dataset)

                train_progress.update(1)
                if i != train_length-1: wandb.log({'Acc': corrects/len(inputs)*100, 'Loss': loss.item()})
                print(f"\rEpoch [{epoch+1:2}/{EPOCHS}], Step [{i+1:2}/{train_length}], Acc: {corrects/len(inputs):.6%}, Loss: {loss.item():.6f}", end="")

            print(f"\rEpoch [{epoch+1:2}/{EPOCHS}], Step [{train_length}/{train_length}], Acc: {train_acc:.6%}, Loss: {train_loss:.6f}", end="")
            val_acc, val_loss = 0, 0

            # Validation
            model.eval()
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)

                    val_loss += criterion(outputs, targets).item() / valid_length
                    val_acc += (torch.max(outputs, 1)[1] == targets.data).sum() / (len(inputs) * valid_length)
                    valid_progress.update(1)

            update(train_loss=train_loss, valid_loss=val_loss)
            wandb.log({'Train Acc': train_acc*100, 'Train Loss': train_loss, 'Val Acc': val_acc*100, 'Val Loss': val_loss})
            print(f"\rEpoch [{epoch+1:2}/{EPOCHS}], Step [{train_length}/{train_length}], Acc: {train_acc:.6%}, Loss: {train_loss:.6f}, Valid Acc: {val_acc:.6%}, Valid Loss: {val_loss:.6f}", end="\n" if (epoch+1) % 5 == 0 or (epoch+1) == EPOCHS else "")
    return (
        corrects,
        epoch,
        epochs,
        i,
        inputs,
        loss,
        outputs,
        targets,
        train_acc,
        train_length,
        train_loss,
        train_progress,
        update,
        val_acc,
        val_loss,
        valid_length,
        valid_progress,
    )


@app.cell
def _(model, path, torch):
    if not path.isdir(path.join(".", "models")):
        import os
        os.mkdir(path.join(".", "models"))

    # Model Save
    save_path = path.join(".", "models", f"normal_vit_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return os, save_path


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""# Test""")
    return


@app.cell
def _(device, model, path, torch):
    # Load Model
    model_id = "normal_vit_model"

    model.load_state_dict(torch.load(path.join(".", "models", f"{model_id}.pt")))
    model.to(device)
    return (model_id,)


@app.cell
def _(device, model, test_dataset, test_loader, torch, tqdm):
    corrects = 0
    test_length = len(test_dataset)

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += (preds == targets.data).sum()
            print(f"Model Accuracy: {corrects/test_length:%}", end="\r")
    return corrects, inputs, outputs, preds, targets, test_length


@app.cell
def _(device, model, test_dataset, test_loader, torch, tqdm):
    corrects = 0
    test_length = len(test_dataset)

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(pixel_values=inputs, labels=targets, use_cache=False)
            _, preds = torch.max(outputs.logits[:, -1, :], 1)
            corrects += (preds == targets.data).sum()
            print(f"Model Accuracy: {corrects/test_length:%}", end="\r")
    return corrects, inputs, outputs, preds, targets, test_length


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
