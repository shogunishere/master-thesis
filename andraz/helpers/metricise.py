import numpy as np
from torchmetrics import JaccardIndex

from andraz import settings


class Metricise:
    def __init__(self, metrics):
        self.names = metrics
        self.metrics = None
        self.reset_metrics()

    def _aggregate_metrics(self):
        for x in self.metrics:
            if x == "Image":
                continue
            self.metrics[x] = np.mean(self.metrics[x])

    def reset_metrics(self):
        self.metrics = {x: [] for x in self.names}

    def add_loss(self, value, name):
        self.metrics["Loss/{}".format(name)].append(value)

    def add_jaccard(self, y, y_pred, name, classes=["back", "weeds", "lettuce"]):
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                self.metrics["Jaccard/{}/{}".format(name, classes[j])].append(
                    self._jaccard(y[i][j], y_pred[i][j])
                )

    def _jaccard(self, y, y_pred, device="cuda:0"):
        jaccard = JaccardIndex("binary").to(device)
        return jaccard(y, y_pred).cpu()

    def add_image(self, image):
        self.metrics["Image"].append(image)

    def report_wandb(self, wandb):
        if settings.WANDB:
            self._aggregate_metrics()
            wandb.log(self.metrics)
            self.reset_metrics()
