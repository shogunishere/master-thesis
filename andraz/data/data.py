import os

import numpy as np
import torch
from torch import Generator, tensor, argmax, ones, zeros, cat
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt

import andraz.settings as settings


class ImageDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]


class ImageImporter:
    def __init__(self, dataset):
        assert dataset in ["agriadapt", "cofly", "infest"]
        self._dataset = dataset

    def get_dataset(self):
        if self._dataset == "agriadapt":
            return self._get_agriadapt()
        elif self._dataset == "cofly":
            return self._get_cofly()
        elif self._dataset == "infest":
            return self._get_infest()

    def _get_agriadapt(self):
        """
        This method only returns raw images as there are no labelled masks for this data for now.
        NOTE: There's a lot of images, if we don't batch import this, RAM will not be happy.
        """
        images = sorted(
            os.listdir(settings.PROJECT_DIR + "data/agriadapt/UAV_IMG/UAV_IMG/")
        )
        create_tensor = transforms.ToTensor()
        smaller = transforms.Resize((1280, 720))
        X, y = [], []
        for file_name in images:
            tens = smaller(
                create_tensor(
                    Image.open(
                        settings.PROJECT_DIR
                        + "data/agriadapt/UAV_IMG/UAV_IMG/"
                        + file_name
                    )
                )
            )
            X.append(tens)
            y.append(tens)
        return ImageDataset(X, y)

    def _get_cofly(self):
        """
        Import images and their belonging segmentation masks (one-hot encoded).
        """
        images = sorted(os.listdir(settings.PROJECT_DIR + "data/cofly/images/images/"))
        create_tensor = transforms.ToTensor()

        # If you want to do any other transformations
        smaller = transforms.Resize((128, 128))

        X, y = [], []
        for file_name in images:
            X.append(
                create_tensor(
                    Image.open(
                        settings.PROJECT_DIR + "data/cofly/images/images/" + file_name
                    )
                )
            )
            # A story behind the beauty below:
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
                                settings.PROJECT_DIR
                                + "data/cofly/labels/labels/"
                                + file_name
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

    def _get_infest(self):
        """
        Import images and convert labels coordinates to actual masks and return a train/test split datasets.
        There are two classes in the segmentation mask labels:
        0 -> weeds
        1 -> lettuce
        The indidces of the sementation mask are 1 and 2 respectively.
        Therefore, we create a 3-channel segmentation mask that separately recognises both weeds and lettuce.
        """
        images = sorted(
            os.listdir(
                settings.PROJECT_DIR
                + "data/agriadapt/NN_labeled_samples_salad_infesting_plants.v1i.yolov7pytorch/test/images/"
            )
        )
        create_tensor = transforms.ToTensor()
        X, y = [], []

        for file_name in images:
            tens = create_tensor(
                Image.open(
                    settings.PROJECT_DIR
                    + "data/agriadapt/NN_labeled_samples_salad_infesting_plants.v1i.yolov7pytorch/test/images/"
                    + file_name
                )
            )
            X.append(tens)
            image_width = tens.shape[1]
            image_height = tens.shape[2]

            # Constructing the segmentation mask
            # We init the whole tensor as background
            y = cat(
                (
                    ones(1, image_width, image_height),
                    zeros(2, image_width, image_height),
                ),
                0,
            )
            # Then, label by label, add to other classes and remove from background.
            with open(
                settings.PROJECT_DIR
                + "data/agriadapt/NN_labeled_samples_salad_infesting_plants.v1i.yolov7pytorch/test/labels/"
                + file_name.replace("jpg", "txt")
            ) as rows:
                labels = [row.rstrip() for row in rows]
                for label in labels:
                    class_id, pixels = self._yolov7_label(
                        label, image_width, image_height
                    )
                    print(class_id)

            0 / 0

        return ImageDataset(X, y)

    def _yolov7_label(self, label, image_width, image_height):
        """
        Implement an image mask generation according to this:
        https://roboflow.com/formats/yolov7-pytorch-txt
        """
        class_id, center_x, center_y, width, height = label.split(" ")
        # TODO: implement returning pixels that correspond to given label (follow the link in the description)
        return int(class_id) + 1, 0


if __name__ == "__main__":
    ii = ImageImporter("infest")
    test = ii.get_dataset()
    X, y = next(iter(test))
    smaller = transforms.Resize((1280, 720))
    X = smaller(X)
    X = X.permute(1, 2, 0)
    plt.imshow(X)
    plt.show()

    # ii = ImageImporter("cofly")
    # train, test = ii.get_dataset()
    # X, y = next(iter(train))
    # y = argmax(y, dim=0)
    # plt.imshow(y)
    # plt.show()
