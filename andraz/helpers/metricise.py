import numpy as np
import torch
import wandb
from numpy import unique
from sklearn.metrics import precision_score
from torchmetrics import JaccardIndex
from torchmetrics.classification import BinaryPrecision

from andraz import settings


class Metricise:
    def __init__(self, metrics):
        self.names = metrics
        self.metrics = None
        self.reset_metrics()

    def _aggregate_metrics(self):
        remove_list = []
        for x in self.metrics:
            if x == "Image":
                if self.metrics[x]:
                    self.metrics[x] = self.metrics[x]
                else:
                    remove_list.append(x)
            elif x == "learning rate":
                self.metrics[x] = self.metrics[x][0]
            else:
                self.metrics[x] = np.mean(self.metrics[x])
        for x in remove_list:
            self.metrics.pop(x)

    def reset_metrics(self):
        self.metrics = {x: [] for x in self.names}

    # Adding metrics
    def add_loss(self, value, name):
        self.metrics["Loss/{}".format(name)].append(value)

    def add_jaccard(self, y, y_pred, name, classes=["back", "weeds", "lettuce"]):
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                self.metrics["Jaccard/{}/{}".format(name, classes[j])].append(
                    self._jaccard(y[i][j], y_pred[i][j])
                )

    def add_precision_old(self, y, y_pred, name, classes=["back", "weeds", "lettuce"]):
        # Iterate through the batch of data
        for i in range(y.shape[0]):
            # Iterate through classes
            for j in range(y.shape[1]):
                # If everything is predicted as negative
                if len(unique(y_pred[i][j].cpu())) == 1:
                    self.metrics["OldPrecision/{}/{}".format(name, classes[j])].append(
                        0
                    )
                else:
                    self.metrics["OldPrecision/{}/{}".format(name, classes[j])].append(
                        # For the binary precision evaluation to work, we need 1D array
                        precision_score(
                            [
                                int(xx)
                                for x in y[i][j].cpu().numpy().tolist()
                                for xx in x
                            ],
                            [
                                int(xx)
                                for x in y_pred[i][j].cpu().numpy().tolist()
                                for xx in x
                            ],
                        )
                    )

    def add_precision(self, y, y_pred, name, classes=["back", "weeds", "lettuce"]):
        precision_calculation = BinaryPrecision(validate_args=False).to("cuda:0")
        for j in range(y.shape[1]):
            result = float(precision_calculation(y[:, j], y_pred[:, j]).cpu())
            self.metrics["Precision/{}/{}".format(name, classes[j])].append(result)

    def add_image(self, X, y, y_pred, epoch):
        y = torch.argmax(y, dim=1)
        y = y + 1

        y_pred = torch.argmax(y_pred, dim=1)
        y_pred = y_pred + 1

        table = wandb.Table(columns=["ID", "Epoch", "Image"])
        class_labels = {1: "Background", 2: "Weeds", 3: "Lettuce"}
        for i in range(10):
            try:
                original_image = X[i].cpu().permute(1, 2, 0)

                mask_org = y[i].cpu()
                mask_pred = y_pred[i].cpu()
                masks = wandb.Image(
                    original_image.numpy(),
                    masks={
                        "labels": {
                            "mask_data": mask_org.numpy(),
                            "class_labels": class_labels,
                        },
                        "predictions": {
                            "mask_data": mask_pred.numpy(),
                            "class_labels": class_labels,
                        },
                    },
                )

                table.add_data(i, epoch, masks)
            except IndexError:
                pass

        self.metrics["Image"] = table

    def add_learning_rate(self, learning_rate):
        self.metrics["learning_rate"].append(learning_rate)

    # Metrics calculation
    def _jaccard(self, y, y_pred, device="cuda:0"):
        jaccard = JaccardIndex("binary").to(device)
        return jaccard(y, y_pred).cpu()

    # Wandb handling
    def report_wandb(self, wandb, epoch=-1):
        if settings.WANDB:
            self._aggregate_metrics()
            wandb.log(self.metrics)
            self.reset_metrics()
