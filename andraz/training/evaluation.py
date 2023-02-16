from datetime import datetime
from time import sleep

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import argmax
from torch.cuda import memory_summary, mem_get_info
from torch.utils.data import DataLoader
from torchvision.utils import save_image, draw_segmentation_masks

import settings
from data.data import get_dataset


def save_images(X, y, pred, batch, width, first):
    for i, (x, test, y) in enumerate(zip(X, y, pred)):
        # Generate an original rgb image with predicted mask overlay.
        x_mask = torch.tensor(
            torch.mul(x.clone().detach().cpu(), 255), dtype=torch.uint8
        )
        mask = torch.tensor(argmax(y.clone().detach(), dim=0), dtype=torch.bool)
        image = draw_segmentation_masks(x_mask, mask, colors="yellow", alpha=0.5)
        plt.imshow(image.permute(1, 2, 0))
        plt.savefig("plots/slim_net/{}_{}_{}_overlay.png".format(batch, i, width))

        x = x.cpu().float().permute(1, 2, 0)
        test = argmax(test, dim=0).float()
        y = argmax(y, dim=0).cpu().float()

        # Generate separate images
        # plt.imshow(x)
        # plt.savefig("plots/{}_{}_org.png".format(batch, i))
        if first:
            plt.imshow(test)
            plt.savefig("plots/slim_net/{}_{}_0_mask.png".format(batch, i))
        # plt.imshow(y)
        # plt.savefig("plots/{}_{}_pred.png".format(batch, i))


def report_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print("Total: {}".format(t / 1024 / 1024))
    print("Reserved: {}".format(r / 1024 / 1024))
    print("Allocated: {}".format(a / 1024 / 1024))
    print()


if __name__ == "__main__":
    model = torch.load("garage/slim_model_100.pt")
    model.eval()
    train, test = get_dataset()
    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    device = "cuda:0"

    X_res, y_test, y_res = [], [], []
    i = 0
    with torch.no_grad():
        for X, y in test_loader:
            first = True
            X = X.to(device)
            torch.cuda.empty_cache()
            w = 1
            for width_mult in settings.width_mult_list:
                param_count = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                model.set_width(width_mult)
                print(width_mult)
                images = model.forward(X)
                # save_images(X, y, images, i, w, first)
                first = False
                w += 1
                print("=================")
            i += 1
            0 / 0
