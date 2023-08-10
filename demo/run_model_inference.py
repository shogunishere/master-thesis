# In short, what we need is a regular U-Net and squeeze U-Net trained on the CoFly-WeedDB dataset
# (using the methodology from the paper, with all weed classes grouped together in a single “Weed”
# label). Ideally, for both networks we should have a basic test bench script that would load the
# network, perform inference on the test set, and report both IoU and precision.
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from torchmetrics.classification import BinaryPrecision

from andraz import settings
from andraz.data.data import ImageImporter
from andraz.helpers.drive_fetch import setup_env


class Inference:
    def __init__(self, model, image_resolution=None):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = self._load_test_model(model)
        self.model.eval()
        self.image_resolution = image_resolution
        self.test_loader = self._load_test_data()
        self.results = {
            x: {"iou": {"back": 0, "weeds": 0}, "precision": {"back": 0, "weeds": 0}}
            for x in settings.WIDTHS
        }

    def _load_test_model(self, model):
        return torch.load(Path(settings.PROJECT_DIR) / "training/garage/" / model)

    def _load_test_data(self):
        ii = ImageImporter(
            "cofly",
            only_test=True,
            smaller=self.image_resolution,
        )
        _, test = ii.get_dataset()
        return DataLoader(test, batch_size=1, shuffle=False)

    def _infer(self):
        with torch.no_grad():
            for width_mult in settings.WIDTHS:
                self.model.set_width(width_mult)
                back_iou = []
                weed_iou = []
                back_precision = []
                weed_precision = []
                for X, y in self.test_loader:
                    X = X.to("cuda:0")
                    y = y.to("cuda:0")
                    y_pred = self.model.forward(X)
                    y_pred = torch.where(y_pred < 0.5, 0, 1)
                    back_iou.append(self._calculate_iou(y[0][0], y_pred[0][0]))
                    weed_iou.append(self._calculate_iou(y[0][1], y_pred[0][1]))
                    back_precision.append(
                        self._calculate_precision(y[0][0], y_pred[0][0])
                    )
                    weed_precision.append(
                        self._calculate_precision(y[0][1], y_pred[0][1])
                    )
                self.results[width_mult]["iou"]["back"] = np.mean(back_iou)
                self.results[width_mult]["iou"]["weeds"] = np.mean(weed_iou)
                self.results[width_mult]["precision"]["back"] = np.mean(back_precision)
                self.results[width_mult]["precision"]["weeds"] = np.mean(weed_precision)
        return self.results

    def _calculate_iou(self, y, y_pred):
        jaccard = JaccardIndex("binary").to(self.device)
        return jaccard(y, y_pred).cpu()

    def _calculate_precision(self, y, y_pred):
        precision_calculation = BinaryPrecision(validate_args=False).to(self.device)
        results = []
        for j in range(y.shape[1]):
            results.append(float(precision_calculation(y[:, j], y_pred[:, j]).cpu()))
        return results

    def run(self):
        self._infer()


if __name__ == "__main__":
    # Download the Cofly dataset and place it in a proper directory.
    # You only have to do this the first time, afterwards the data is ready to go.
    setup_env()

    # Select a model from andraz/training/garage directory and set the
    # image resolution tuple to match the image input size of the model
    for size in [128, 256, 512]:
        infer = Inference(
            "cofly_slim_{}.pt".format(size), image_resolution=(size, size)
        )

        infer.run()

        # Results are stored in a dictionary
        print(infer.results)
        print()
