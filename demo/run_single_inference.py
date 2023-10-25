import os
from pathlib import Path
from random import randint

from PIL import Image
from matplotlib import pyplot as plt

import torch
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks

from andraz import settings
from andraz.data.data import ImageImporter
from andraz.helpers.drive_fetch import setup_env
from andraz.helpers.metricise import Metricise
from ioana.inference import AdaptiveWidth


class SingleImageInference:
    def __init__(
        self,
        dataset,
        image_resolution,
        model_architecture,
        fixed_image=-1,
        save_image=False,
        is_trans=False,
        is_best_fitting=False,
    ):
        self.project_path = Path(settings.PROJECT_DIR)
        if dataset == "infest":
            self.image_dir = "andraz/data/agriadapt/NN_labeled_samples_salad_infesting_plants.v1i.yolov7pytorch/test/images/"
        elif dataset == "geok":
            self.image_dir = "andraz/data/geok/test/images/"
        else:
            raise ValueError("Invalid dataset selected.")
        assert model_architecture in ["slim", "squeeze"]
        self.model_architecture = model_architecture
        self.image_resolution = image_resolution
        model_key = f"{dataset}_{model_architecture}_{image_resolution[0]}"
        if is_trans:
            model_key += "_trans"
        if is_best_fitting:
            model_key += "_opt"
        self.model = torch.load(
            Path(settings.PROJECT_DIR) / f"andraz/training/garage/{model_key}.pt"
        )
        self.fixed_image = fixed_image
        self.save_image = save_image
        self.adaptive_width = AdaptiveWidth(model_key)
        self.tensor_to_image = ImageImporter(dataset).tensor_to_image
        self.random_image_index = -1

    def _get_random_image_path(self):
        images = os.listdir(self.project_path / self.image_dir)
        if self.fixed_image < 0:
            self.random_image_index = randint(0, len(images) - 1)
            return images[self.random_image_index]
        else:
            return images[
                self.fixed_image if self.fixed_image < len(images) else len(images) - 1
            ]

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

        return int(class_id), pixels

    def _get_single_image(self):
        file_name = self._get_random_image_path()
        img = Image.open(self.project_path / self.image_dir / file_name)
        create_tensor = transforms.ToTensor()
        smaller = transforms.Resize(self.image_resolution)

        img = smaller(img)
        img = create_tensor(img)

        image_width = img.shape[1]
        image_height = img.shape[2]

        # Constructing the segmentation mask
        # We init the whole tensor as the background
        mask = torch.cat(
            (
                torch.ones(1, image_width, image_height),
                torch.zeros(1, image_width, image_height),
            ),
            0,
        )
        # Then, label by label, add to other classes and remove from background.
        file_name = file_name[:-3] + "txt"
        with open(
            self.project_path / self.image_dir.replace("images", "labels") / file_name
        ) as rows:
            labels = [row.rstrip() for row in rows]
            for label in labels:
                class_id, pixels = self._yolov7_label(label, image_width, image_height)
                if class_id != 1:
                    continue
                # Change values based on received pixels
                for pixel in pixels:
                    mask[0][pixel[0]][pixel[1]] = 0
                    mask[class_id][pixel[0]][pixel[1]] = 1

        img = img.to("cuda:0")
        mask = mask.to("cuda:0")
        img = img[None, :]
        mask = mask[None, :]

        return img, mask

    def _generate_images(self, X, y, y_pred):
        # Generate an original rgb image with predicted mask overlay.
        x_mask = torch.tensor(
            torch.mul(X.clone().detach().cpu(), 255), dtype=torch.uint8
        )
        x_mask = x_mask[0]

        # Draw predictions
        y_pred = y_pred[0]
        mask = torch.argmax(y_pred.clone().detach(), dim=0)
        weed_mask = torch.where(mask == 1, True, False)[None, :, :]
        # lettuce_mask = torch.where(mask == 2, True, False)[None, :, :]
        # mask = torch.cat((weed_mask, lettuce_mask), 0)

        image = draw_segmentation_masks(x_mask, weed_mask, colors=["red"], alpha=0.5)
        plt.imshow(image.permute(1, 2, 0))
        plt.savefig(f"results/{self.random_image_index}_pred.jpg")

        # Draw ground truth
        mask = y.clone().detach()[0]
        weed_mask = torch.where(mask[1] == 1, True, False)[None, :, :]
        # lettuce_mask = torch.where(mask[2] == 1, True, False)[None, :, :]
        # mask = torch.cat((weed_mask, lettuce_mask), 0)
        image = draw_segmentation_masks(x_mask, weed_mask, colors=["red"], alpha=0.5)
        plt.imshow(image.permute(1, 2, 0))
        plt.savefig(f"results/{self.random_image_index}_true.jpg")

    def infer(self, fixed=-1):
        # Get a random single image from test dataset.
        # Set the fixed attribute to always obtain the same image
        image, mask = self._get_single_image()

        # Select and set the model width
        width = self.adaptive_width.get_image_width(
            self.tensor_to_image(image.cpu())[0]
        )
        self.model.set_width(width)

        # Get a prediction
        y_pred = self.model.forward(image)

        metrics = Metricise()
        metrics.calculate_metrics(mask, y_pred, "test")
        results = metrics.report(None)

        # Generate overlayed segmentation masks (ground truth and prediction)
        if self.save_image:
            self._generate_images(image, mask, y_pred)

        return results


if __name__ == "__main__":
    # Run this once to download the new dataset
    # setup_env()

    si = SingleImageInference(
        # geok (new dataset) or infest (forbidden dataset)
        dataset="geok",
        # Tuple of two numbers: (128, 128), (256, 256), or (512, 512)
        image_resolution=(
            256,
            256,
        ),
        # slim or squeeze
        model_architecture="slim",
        # Set to a positive integer to select a specific image from the dataset, otherwise random
        fixed_image=-1,
        # Do you want to generate a mask/image overlay
        save_image=True,
        # Was segmentation model trained using transfer learning
        is_trans=True,
        # Was segmentation model trained with find_best_fitting (utilising
        # model that has the highest difference in iou between widths
        is_best_fitting=False,
    )
    for i in range(10):
        results = si.infer()
        print(results)
