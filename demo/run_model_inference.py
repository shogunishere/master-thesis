# In short, what we need is a regular U-Net and squeeze U-Net trained on the CoFly-WeedDB dataset
# (using the methodology from the paper, with all weed classes grouped together in a single “Weed”
# label). Ideally, for both networks we should have a basic test bench script that would load the
# network, perform inference on the test set, and report both IoU and precision.
import os
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from plotly import graph_objects as go
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryJaccardIndex,
)
from torchvision.transforms import transforms

from andraz import settings
from andraz.data.data import ImageImporter
from andraz.helpers.drive_fetch import setup_env
from ioana.inference import AdaptiveWidth

METRICS = {
    "iou": BinaryJaccardIndex,
    "precision": BinaryPrecision,
    "recall": BinaryRecall,
    "f1score": BinaryF1Score,
}


class Inference:
    def __init__(
        self,
        model,
        metrics=None,
        image_resolution=None,
        create_images=False,
    ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_name = model.split(".")[0]
        self.create_images = create_images
        self.model = self._load_test_model(model)
        self.model.eval()
        self.image_resolution = image_resolution
        self.metrics = METRICS if metrics is None else metrics
        self.results = {
            x: {
                "iou": {"back": [], "weeds": []},
                "precision": {"back": [], "weeds": []},
                "recall": {"back": [], "weeds": []},
                "f1score": {"back": [], "weeds": []},
            }
            for x in settings.WIDTHS + ["adapt"]
        }
        self.width_distribution = {x: 0 for x in settings.WIDTHS}
        self.width_selection = AdaptiveWidth()
        self.tensor_to_image = ImageImporter("cofly").tensor_to_image

    @staticmethod
    def _load_test_model(model):
        return torch.load(
            Path(settings.PROJECT_DIR) / "andraz/training/garage/" / model
        )

    def _load_test_data(self, smaller):
        ii = ImageImporter(
            "cofly",
            only_test=True,
            smaller=smaller,
        )
        _, test = ii.get_dataset()
        return DataLoader(test, batch_size=1, shuffle=False)

    def _infer(self):
        # Infer for all available widths of the model
        with torch.no_grad():
            test_loader = self._load_test_data(self.image_resolution)
            for width in settings.WIDTHS:
                self.model.set_width(width)
                i = 1
                for X, y in test_loader:
                    X = X.to("cuda:0")
                    y = y.to("cuda:0")
                    y_pred = self.model.forward(X)
                    y_pred = torch.where(y_pred < 0.5, 0, 1)

                    if self.create_images:
                        self._generate_mask_image(
                            X[0].permute(1, 2, 0).cpu(),
                            y[0][1].cpu(),
                            y_pred[0][1].cpu(),
                            int(width * 100),
                            i,
                        )

                    for metric in self.metrics:
                        for j, pred_class in enumerate(["back", "weeds"]):
                            self.results[width][metric][pred_class].append(
                                self.metrics[metric](validate_args=False)
                                .to(self.device)(y[0][j], y_pred[0][j])
                                .cpu()
                            )
        # Infer with the adaptation algorithm
        with torch.no_grad():
            # We load the test loader again, as we need full-size
            # images in order for the width selection to work properly
            test_loader = self._load_test_data(False)
            smaller = transforms.Resize(self.image_resolution)
            i = 1
            for X, y in test_loader:
                # Get the image width and set the model to it
                image = self.tensor_to_image(X)[0]
                width = self.width_selection.get_image_width(image)
                self.model.set_width(width)
                self.width_distribution[width] += 1
                # Resize the image to the required input size
                X = smaller(X)
                y = smaller(y)

                X = X.to("cuda:0")
                y = y.to("cuda:0")
                y_pred = self.model.forward(X)
                y_pred = torch.where(y_pred < 0.5, 0, 1)

                if self.create_images:
                    self._generate_mask_image(
                        X[0].permute(1, 2, 0).cpu(),
                        y[0][1].cpu(),
                        y_pred[0][1].cpu(),
                        "adapt",
                        i,
                    )

                for metric in self.metrics:
                    for j, pred_class in enumerate(["back", "weeds"]):
                        self.results["adapt"][metric][pred_class].append(
                            self.metrics[metric](validate_args=False)
                            .to(self.device)(y[0][j], y_pred[0][j])
                            .cpu()
                        )
        self._average_results()
        return self.results

    def _average_results(self):
        for width_mult in settings.WIDTHS + ["adapt"]:
            for metric in self.metrics:
                for j, pred_class in enumerate(["back", "weeds"]):
                    self.results[width_mult][metric][pred_class] = np.mean(
                        self.results[width_mult][metric][pred_class]
                    )

    def _generate_mask_image(self, x, y, pred, width, i):
        save_path = "images/{}".format(self.model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if width == 25:
            plt.imshow(x)
            plt.savefig("{}/{}_{}.png".format(save_path, str(i).zfill(4), "image"))
            plt.imshow(y)
            plt.savefig("{}/{}_{}.png".format(save_path, str(i).zfill(4), "mask"))
        plt.imshow(pred)
        plt.savefig("{}/{}_{}_{}.png".format(save_path, str(i).zfill(4), "pred", width))

    def run(self):
        self._infer()


class Comparator:
    def __init__(self, models, metrics=None):
        self.models = models
        self.metrics = METRICS if metrics is None else metrics

    def _draw_tab(self, results):
        for pred_class in ["back", "weeds"]:
            for metric in self.metrics:
                print(
                    "======================================================================================"
                )
                print(metric, pred_class)
                print()
                print("{:6s}".format(""), end="")
                for model, _ in self.models:
                    print("{:20s}".format(model), end=" ")
                print()
                for width in settings.WIDTHS + ["adapt"]:
                    print("{:5s}".format(str(width)), end=" ")
                    for model, _ in self.models:
                        print(
                            "{:20s}".format(
                                str(
                                    round(
                                        results[model][width][metric][pred_class] * 100,
                                        2,
                                    )
                                )
                                + " %"
                            ),
                            end=" ",
                        )
                    print()
                print()

    def run(self):
        # Select a model from andraz/training/garage directory and set the
        # image resolution tuple to match the image input size of the model
        results = {}
        graph = True
        for model, size in self.models:
            infer = Inference(
                model,
                image_resolution=(size, size),
                create_images=False,
            )
            infer.run()
            results[model] = infer.results

            if graph:
                x = [k for k in infer.width_distribution]
                y = [v for _, v in infer.width_distribution.items()]
                fig = go.Figure(
                    data=go.Bar(
                        x=x,
                        y=y,
                    )
                )
                fig.update_layout(
                    {"title": "Distribution of selected widths for test images."}
                )
                fig.update_xaxes(type="category")
                fig.show()
                # Only show once as all the plots will be the same
                graph = False

        self._draw_tab(results)


if __name__ == "__main__":
    # Download the Cofly dataset and place it in a proper directory.
    # You only have to do this the first time, afterwards the data is ready to go.
    # setup_env()

    # A tuple of model name and image input size
    models = [("cofly_slim_{}.pt".format(size), size) for size in [128, 256, 512]] + [
        ("cofly_squeeze_{}.pt".format(size), size) for size in [128, 256, 512]
    ]
    comparator = Comparator(models)

    # Run inference for multiple models and display comparative tables
    comparator.run()
