import numpy as np
from numpy import unique
from sklearn.metrics import precision_score
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
            elif x == "learning rate":
                self.metrics[x] = self.metrics[x][0]
            else:
                self.metrics[x] = np.mean(self.metrics[x])

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

    def add_precision(self, y, y_pred, name, classes=["back", "weeds", "lettuce"]):
        # Iterate through the batch of data
        for i in range(y.shape[0]):
            # Iterate through classes
            for j in range(y.shape[1]):
                # If everything is predicted as negative
                if len(unique(y_pred[i][j].cpu())) == 1:
                    self.metrics["Precision/{}/{}".format(name, classes[j])].append(0)
                else:
                    self.metrics["Precision/{}/{}".format(name, classes[j])].append(
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

    def add_image(self, image):
        self.metrics["Image"].append(image)

    def add_learning_rate(self, learning_rate):
        self.metrics["learning_rate"].append(learning_rate)

    # Metrics calculation
    def _jaccard(self, y, y_pred, device="cuda:0"):
        jaccard = JaccardIndex("binary").to(device)
        return jaccard(y, y_pred).cpu()

    # Wandb handling
    def report_wandb(self, wandb):
        if settings.WANDB:
            self._aggregate_metrics()
            wandb.log(self.metrics)
            self.reset_metrics()
