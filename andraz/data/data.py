import os
import random
from pathlib import Path

import numpy as np
import torch
import cv2
from torch import Generator, tensor, argmax, ones, zeros, cat, unique
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks

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
    def __init__(
        self,
        dataset,
        sample=False,
        validation=False,
        smaller=False,
        only_test=False,
        augmentations=None,
    ):
        assert dataset in ["agriadapt", "cofly", "infest"]
        self._dataset = dataset
        # Only take 10 images per set.
        self.sample = sample
        # If True, return validation instead of testing set (where applicable)
        self.validation = validation
        # Make the images smaller
        self.smaller = smaller
        # Only return the test dataset (first part of returned tuple empty)
        self.only_test = only_test

        self.project_path = Path(settings.PROJECT_DIR)

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
            os.listdir(self.project_path / "data/agriadapt/UAV_IMG/UAV_IMG/")
        )
        create_tensor = transforms.ToTensor()
        smaller = transforms.Resize((1280, 720))
        X, y = [], []
        for file_name in images:
            tens = smaller(
                create_tensor(
                    Image.open(
                        self.project_path
                        / "data/agriadapt/UAV_IMG/UAV_IMG/"
                        / file_name
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
        images = sorted(
            os.listdir(self.project_path / "/home/agriadapt/agriadapt/andraz/data/cofly/images/images/")
        )
        random.seed(42069)
        idx = [x for x in range(len(images))]
        random.shuffle(idx)
        cut = int(len(images) * 0.8)
        train_images = [images[x] for x in idx[:cut]]
        test_images = [images[x] for x in idx[cut:]]
        return self._get_cofly_train(train_images), self._get_cofly_test(test_images)

    def _get_cofly_train(self, images):
        create_tensor = transforms.ToTensor()
        if self.smaller:
            smaller = transforms.Resize(self.smaller)

        X, y = [], []

        for file_name in images:
            img = create_tensor(
                Image.open(
                    self.project_path / "/home/agriadapt/agriadapt/andraz/data/cofly/images/images/" / file_name
                )
            )
            if self.smaller:
                img = smaller(img)

            # Data augmentation
            imgh = transforms.RandomHorizontalFlip(p=1)(img)
            imgv = transforms.RandomVerticalFlip(p=1)(img)
            imghv = transforms.RandomVerticalFlip(p=1)(imgh)

            X.append(img)
            X.append(imgh)
            X.append(imgv)
            X.append(imghv)

            # Open the mask
            mask = Image.open(
                self.project_path / "/home/agriadapt/agriadapt/andraz/data/cofly/labels/labels/" / file_name
            )
            if self.smaller:
                mask = smaller(mask)
            mask = tensor(np.array(mask))
            # Merge weeds classes to a single weeds class
            mask = torch.where(
                mask > 0,
                1,
                0,
            )

            maskh = transforms.RandomHorizontalFlip(p=1)(mask)
            maskv = transforms.RandomVerticalFlip(p=1)(mask)
            maskhv = transforms.RandomVerticalFlip(p=1)(maskh)
            y.append(self._cofly_prep_mask(mask))
            y.append(self._cofly_prep_mask(maskh))
            y.append(self._cofly_prep_mask(maskv))
            y.append(self._cofly_prep_mask(maskhv))

        return ImageDataset(X, y)

    def _get_cofly_test(self, images):
        create_tensor = transforms.ToTensor()
        if self.smaller:
            smaller = transforms.Resize(self.smaller)

        X, y = [], []

        for file_name in images:
            img = create_tensor(
                Image.open(
                    self.project_path / "/home/agriadapt/agriadapt/andraz/data/cofly/images/images/" / file_name
                )
            )
            if self.smaller:
                img = smaller(img)

            X.append(img)

            # Open the mask
            mask = Image.open(
                self.project_path / "/home/agriadapt/agriadapt/andraz/data/cofly/labels/labels/" / file_name
            )
            if self.smaller:
                mask = smaller(mask)
            mask = tensor(np.array(mask))
            # Merge weeds classes to a single weeds class
            mask = torch.where(
                mask > 0,
                1,
                0,
            )
            y.append(self._cofly_prep_mask(mask))

        return ImageDataset(X, y)

    def tensor_to_image(self, tensor_images):
        images = []
        for elem in tensor_images:
            elem = (elem.numpy() * 255).astype(np.uint8)
            elem = elem.transpose(1, 2, 0)
            image = cv2.cvtColor(elem, cv2.COLOR_RGB2BGR)
            images.append(image)
        return images

    def _cofly_prep_mask(self, mask):
        return (
            torch.nn.functional.one_hot(
                mask,
                num_classes=2,
            )
            .permute(2, 0, 1)
            .float()
        )

    def _get_infest(self):
        """
        Import images and convert labels coordinates to actual masks and return a train/test split datasets.
        There are two classes in the segmentation mask labels:
        0 -> weeds
        1 -> lettuce
        The indices of the segmentation mask are 1 and 2 respectively.
        Therefore, we create a 3-channel segmentation mask that separately recognises both weeds and lettuce.
        mask[0] -> background
        mask[1] -> weeds
        mask[2] -> lettuce
        """
        if self.validation:
            return self._fetch_infest_split("train"), self._fetch_infest_split("valid")
        else:
            if self.only_test:
                return None, self._fetch_infest_split("test")
            else:
                return self._fetch_infest_split("train"), self._fetch_infest_split(
                    "test"
                )

    def _fetch_infest_split(self, split="train"):
        images = sorted(
            os.listdir(
                self.project_path
                / "/home/agriadapt/agriadapt/andraz/data/agriadapt/NN_labeled_samples_salad_infesting_plants.v1i.yolov7pytorch/"
                / split
                / "images/"
            )
        )
        create_tensor = transforms.ToTensor()
        X, y = [], []

        if self.sample:
            images = images[:10]

        for file_name in images:
            img = Image.open(
                self.project_path
                / "/home/agriadapt/agriadapt/andraz/data/agriadapt/NN_labeled_samples_salad_infesting_plants.v1i.yolov7pytorch/"
                / split
                / "images/"
                / file_name
            )
            if self.smaller:
                smaller = transforms.Resize(self.smaller)
                img = smaller(img)
            tens = create_tensor(img)
            X.append(tens)
            image_width = tens.shape[1]
            image_height = tens.shape[2]
            # image_width = 640
            # image_height = 640

            # Constructing the segmentation mask
            # We init the whole tensor as background
            mask = cat(
                (
                    ones(1, image_width, image_height),
                    zeros(2, image_width, image_height),
                ),
                0,
            )
            # Then, label by label, add to other classes and remove from background.
            with open(
                self.project_path
                / "/home/agriadapt/agriadapt/andraz/data/agriadapt/NN_labeled_samples_salad_infesting_plants.v1i.yolov7pytorch/"
                / split
                / "labels/"
                / file_name.replace("jpg", "txt")
            ) as rows:
                labels = [row.rstrip() for row in rows]
                for label in labels:
                    class_id, pixels = self._yolov7_label(
                        label, image_width, image_height
                    )
                    # Change values based on received pixels
                    for pixel in pixels:
                        mask[0][pixel[0]][pixel[1]] = 0
                        mask[class_id][pixel[0]][pixel[1]] = 1
            y.append(mask)

        return ImageDataset(X, y)

    def _yolov7_label(self, label, image_width, image_height):
        """
        Implement an image mask generation according to this:
        https://roboflow.com/formats/yolov7-pytorch-txt
        """
        # Deconstruct a row
        class_id, center_x, center_y, width, height = [
            float(x) for x in label.split(" ")
        ]

        # Get center pixel
        center_x = center_x * image_width
        center_y = center_y * image_height

        # Get border pixels
        top_border = int(center_x - (width / 2 * image_width))
        bottom_border = int(center_x + (width / 2 * image_width))
        left_border = int(center_y - (height / 2 * image_height))
        right_border = int(center_y + (height / 2 * image_height))

        # Generate pixels
        pixels = []
        for x in range(left_border, right_border):
            for y in range(top_border, bottom_border):
                pixels.append((x, y))

        return int(class_id + 1), pixels


if __name__ == "__main__":
    ii = ImageImporter("cofly")
    train, test = ii.get_dataset()
    print(len(train))
    print(len(test))

    X, y = next(iter(train))
    x_mask = torch.tensor(torch.mul(X, 255), dtype=torch.uint8)
    mask = torch.tensor(y[1:], dtype=torch.bool)
    image = draw_segmentation_masks(x_mask, mask, colors=["yellow", "green"], alpha=0.5)
    plt.imshow(image.permute(1, 2, 0))
    plt.show()

    # ii = ImageImporter("cofly")
    # train, test = ii.get_dataset()
    # X, y = next(iter(train))
    # y = argmax(y, dim=0)
    # plt.imshow(y)
    # plt.show()
