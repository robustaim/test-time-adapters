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
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
