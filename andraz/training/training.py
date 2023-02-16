from datetime import datetime

import torch
import wandb as wandb
from numpy import unique
from torch import argmax, tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torchvision.transforms.functional as F
import andraz.settings as settings
from andraz.data.data import get_dataset
from andraz.models.slim_unet import SlimUNet


def get_config():
    return {
        "Epochs": settings.EPOCHS,
        "Batch size": settings.BATCH_SIZE,
        "Learning rate": settings.LEARNING_RATE,
    }


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()


def metrics(y, outputs):
    """
    Calculate the % of mask being predicted.
    """
    y, o = y.cpu(), outputs.cpu()
    ynz = torch.nonzero(y)
    onz = torch.nonzero(o)

    # Intersection of non-zero elements -> true positive
    tp = len(np.intersect1d(ynz, onz))

    return round(tp / y.shape[0], 4)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


if __name__ == "__main__":
    if settings.WANDB:
        wandb.init(project="agriadapt", entity="colosal", config={})
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    train, test = get_dataset()
    train_loader = DataLoader(train, batch_size=settings.BATCH_SIZE, shuffle=False)

    loss_function = torch.nn.CrossEntropyLoss(
        weight=tensor([0.16, 0.28, 0, 0.28, 0.28]).to(device)
    )

    # model = SimpleConv(3, 5)
    model = SlimUNet(1, 2)
    model.to(device)
    model.train()

    optimizer = Adam(model.parameters(), lr=settings.LEARNING_RATE)
    first = True

    for epoch in range(settings.EPOCHS):
        losses = []
        se = datetime.now()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            for width_mult in settings.width_mult_list:
                model.set_width(width_mult)
                outputs = model.forward(X)
                loss = loss_function(outputs, y)

                loss.backward()
            optimizer.step()

            # tps.append(metrics(y, outputs))
            losses.append(loss.detach().cpu())

        print("Loss: {}".format(np.mean(losses)))
        print("Epoch Time: {}".format(datetime.now() - se))

        if settings.WANDB:
            wandb.log({"Train/loss": np.mean(losses)})

        if epoch % 10 == 0:
            if first:
                img = argmax(y[0], dim=0).cpu().detach()
                plt.imshow(img)
                plt.show()
                first = False

            img = argmax(outputs[0], dim=0).cpu().detach()
            plt.imshow(img)
            plt.show()
            print(unique(img))
        if epoch % 100 == 0:
            torch.save(model, "slim_model_{}.pt".format(epoch))

        print()
    if settings.WANDB:
        wandb.finish()

    torch.save(model, "slim_model.pt")
