from random import randint
from typing import Any

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def plot_misclassified(
    model: Any,
    data_loader: DataLoader,
    device: torch.device,
    transformations: A.Compose,
    title: str = "Misclassified (pred/ truth)",
    return_imgs: bool = False,
):
    count = 1
    no_misclf: int = 10
    rows, cols = 2, int(no_misclf / 2)
    figure = plt.figure(figsize=(cols * 3, rows * 3))

    classes = data_loader.dataset.classes
    dataset = data_loader.dataset.ds

    model = model.to(device)
    model.eval()
    imgs_list = []
    with torch.inference_mode():
        while True:
            k = randint(0, len(dataset))
            img, label = dataset[k]
            img = np.array(img)

            aug_img = transformations(image=img)["image"]
            pred = model(aug_img.unsqueeze(0).to(device)).argmax().item()  # Prediction
            if pred != label:
                imgs_list.append(img.copy())

                figure.add_subplot(rows, cols, count)  # adding sub plot
                plt.title(f"{classes[pred]} / {classes[label]}")  # title of plot
                plt.axis("off")
                plt.imshow(img)

                count += 1
                if count == no_misclf + 1:
                    break

    plt.suptitle(title, fontsize=15)
    plt.show()

    if return_imgs:
        return imgs_list