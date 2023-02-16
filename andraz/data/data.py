import os

import numpy as np
import torch
from numpy import unique
from torch import Generator, tensor, argmax
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms

from PIL import Image
import matplotlib.pyplot as plt

from plotly import graph_objects as go


import andraz.settings as settings


class ImageDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]


def get_dataset():
    images = sorted(os.listdir(settings.PROJECT_DIR + "data/images/images/"))
    create_tensor = transforms.ToTensor()

    X, y = [], []
    smaller = transforms.Resize((128, 128))
    classes = []
    for file_name in images:
        X.append(
            create_tensor(
                Image.open(settings.PROJECT_DIR + "data/images/images/" + file_name)
            )
        )
        # The story behind the beauty below:
        # As Torch is kinda shady about how to one hot encode segmentation masks, we made our own.
        # We first open the image with PIL, convert it to numpy array, then to Torch tensor,
        # then cast it to long (required for one-hot encoding), then perform one hot encoding,
        # then permuting the dimensions to fit the shape expected by torch while training, then
        # cast everything to float, so it works with our model.
        y.append(
            torch.nn.functional.one_hot(
                tensor(
                    np.array(
                        Image.open(
                            settings.PROJECT_DIR + "data/labels/labels/" + file_name
                        )
                    )
                ).long(),
                num_classes=5,
            )
            .permute(2, 0, 1)
            .float()
        )
    return random_split(
        ImageDataset(X, y),
        [0.75, 0.25],
        generator=Generator().manual_seed(settings.SEED),
    )


if __name__ == "__main__":
    train, test = get_dataset()
    # 0 / 0
    X, y = next(iter(train))
    y = argmax(y, dim=0)
    plt.imshow(y)
    plt.show()
    #
    # y = Image.open(
    #     settings.PROJECT_DIR
    #     + "data/labels/labels/ID_00079_UAV_dji.phantom.4.pro.hawk.1_[Lat=39.5419326810563,Lon=22.643614647699216,Alt=5.0]_DATE_03_07_2019_14_39_44.png"
    # )
    # y = tensor(np.array(y)).long()
    # print(y.shape)
    # y = torch.nn.functional.one_hot(y)
    # print(y.shape)
    # y = y.permute(2, 0, 1)
    # print(y.shape)
    # y = Image.open(
    #     settings.PROJECT_DIR
    #     + "data/labels/labels/ID_00079_UAV_dji.phantom.4.pro.hawk.1_[Lat=39.5419326810563,Lon=22.643614647699216,Alt=5.0]_DATE_03_07_2019_14_39_44.png"
    # )
    # y = transforms.ToTensor()(y)
    # print(y.shape)
    # 0 / 0
    # print(np.array(y).tolist())
    #
    # y = np.array(y)
    # print(y.shape)
    # print(y[400][1000])
    # print(y[600][400])
    # y[y != 0] = 1
    # print(y[400][1000])
    # print(y[600][400])
    # y = transforms.ToTensor()(y)
    # print(y[0][400][1000])
    # print(y[0][600][400])
    # idx = torch.nonzero(y)
    # for x in idx:
    #     y[x[0]][x[1]][x[2]] = 1
    # print(y[0][400][1000])
    # print(y[0][600][400])
    #
    # plt.imshow(y.permute(1, 2, 0))
    # plt.show()
