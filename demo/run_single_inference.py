import os
from pathlib import Path
from random import randint

from PIL import Image
from matplotlib import pyplot as plt

import torch
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks

from andraz import settings


def get_random_image_path(project_path, fixed=-1):
    images = os.listdir(
        project_path
        / "data/agriadapt/NN_labeled_samples_salad_infesting_plants.v1i.yolov7pytorch/test/images/"
    )
    if fixed < 0:
        return images[randint(0, len(images) - 1)]
    else:
        return images[fixed if fixed < len(images) else len(images) - 1]


def _yolov7_label(label, image_width, image_height):
    """
    Implement an image mask generation according to this:
    https://roboflow.com/formats/yolov7-pytorch-txt
    """
    # Deconstruct a row
    class_id, center_x, center_y, width, height = [float(x) for x in label.split(" ")]

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


def get_single_image(fixed=-1):
    project_path = Path(settings.PROJECT_DIR)
    file_name = get_random_image_path(project_path, fixed=fixed)
    img = Image.open(
        project_path
        / "data/agriadapt/NN_labeled_samples_salad_infesting_plants.v1i.yolov7pytorch/test/images/"
        / file_name
    )
    create_tensor = transforms.ToTensor()
    smaller = transforms.Resize((128, 128))

    img = smaller(img)
    img = create_tensor(img)

    image_width = img.shape[1]
    image_height = img.shape[2]

    # Constructing the segmentation mask
    # We init the whole tensor as the background
    mask = torch.cat(
        (
            torch.ones(1, image_width, image_height),
            torch.zeros(2, image_width, image_height),
        ),
        0,
    )
    # Then, label by label, add to other classes and remove from background.
    with open(
        project_path
        / "data/agriadapt/NN_labeled_samples_salad_infesting_plants.v1i.yolov7pytorch/test/labels/"
        / file_name.replace("jpg", "txt")
    ) as rows:
        labels = [row.rstrip() for row in rows]
        for label in labels:
            class_id, pixels = _yolov7_label(label, image_width, image_height)
            # Change values based on received pixels
            for pixel in pixels:
                mask[0][pixel[0]][pixel[1]] = 0
                mask[class_id][pixel[0]][pixel[1]] = 1

    img = img.to("cuda:0")
    mask = mask.to("cuda:0")
    img = img[None, :]
    mask = mask[None, :]

    return img, mask


def save_images(X, y, y_pred):
    # Generate an original rgb image with predicted mask overlay.
    x_mask = torch.tensor(torch.mul(X.clone().detach().cpu(), 255), dtype=torch.uint8)
    x_mask = x_mask[0]

    # Draw predictions
    y_pred = y_pred[0]
    mask = torch.argmax(y_pred.clone().detach(), dim=0)
    weed_mask = torch.where(mask == 1, True, False)[None, :, :]
    lettuce_mask = torch.where(mask == 2, True, False)[None, :, :]
    mask = torch.cat((weed_mask, lettuce_mask), 0)

    image = draw_segmentation_masks(x_mask, mask, colors=["red", "green"], alpha=0.5)
    plt.imshow(image.permute(1, 2, 0))
    plt.savefig("prediction.jpg")

    # Draw ground truth
    mask = y.clone().detach()[0]
    weed_mask = torch.where(mask[1] == 1, True, False)[None, :, :]
    lettuce_mask = torch.where(mask[2] == 1, True, False)[None, :, :]
    mask = torch.cat((weed_mask, lettuce_mask), 0)
    image = draw_segmentation_masks(x_mask, mask, colors=["red", "green"], alpha=0.5)
    plt.imshow(image.permute(1, 2, 0))
    plt.savefig("groundtruth.jpg")


if __name__ == "__main__":
    # Load model and set width
    model = torch.load(Path(settings.PROJECT_DIR) / "training/garage/big_squeeze.pt")
    model.set_width(settings.WIDTHS[3])

    # Get a random single image from test dataset.
    # Set the fixed parameter to always obtain the same image
    image, mask = get_single_image()

    # Get a prediction and generate a ground truth and prediction segmentation masks
    y_pred = model.forward(image)
    save_images(image, mask, y_pred)
