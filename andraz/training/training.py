import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb as wandb
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
from numpy import mean
from torch import tensor, argmax
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, ExponentialLR
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info
import thop
import pthflops
from plotly import graph_objects as go
from torchsummary import summary

import andraz.settings as settings
from andraz.data.data import ImageImporter
from andraz.helpers.masking import get_binary_masks_infest
from andraz.helpers.metricise import Metricise
from andraz.helpers.model_profiling import model_profiling
from andraz.models.slim_squeeze_unet import (
    SlimSqueezeUNet,
    SlimPrunedSqueezeUNet,
    SlimSqueezeUNetCofly,
)
from andraz.models.slim_unet import SlimUNet


class Training:
    def __init__(
        self,
        device,
        architecture=settings.MODEL,
        epochs=settings.EPOCHS,
        learning_rate=settings.LEARNING_RATE,
        learning_rate_scheduler=settings.LEARNING_RATE_SCHEDULER,
        batch_size=settings.BATCH_SIZE,
        regularisation_l2=settings.REGULARISATION_L2,
        image_resolution=settings.IMAGE_RESOLUTION,
        widths=settings.WIDTHS,
        dropout=settings.DROPOUT,
        verbose=1,
        wandb_group=None,
        dataset="infest",
        continue_model="",  # This is set to model name that we want to continue training with (fresh training if "")
        sample=0,
    ):
        self.architecture = architecture
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.learning_rate_scheduler = learning_rate_scheduler
        self.batch_size = batch_size
        self.regularisation_l2 = regularisation_l2
        self.image_resolution = image_resolution
        self.widths = widths
        self.dropout = dropout
        self.verbose = verbose
        self.wandb_group = wandb_group
        self.dataset = dataset
        self.continue_model = continue_model
        self.sample = sample

        self.best_fitting = [0, 0, 0, 0]

    def _report_settings(self):
        print("=======================================")
        print("Training with the following parameters:")
        print("Dataset: {}".format(self.dataset))
        print("Model architecture: {}".format(self.architecture))
        print("Epochs: {}".format(self.epochs))
        print("Learning rate: {}".format(self.learning_rate))
        print("Learning rate scheduler: {}".format(self.learning_rate_scheduler))
        print("Batch size: {}".format(self.batch_size))
        print("L2 regularisation: {}".format(self.regularisation_l2))
        print("Image resolution: {}".format(self.image_resolution))
        print("Dropout: {}".format(self.dropout))
        print("Network widths: {}".format(self.widths))
        print("Loss function weights: {}".format(settings.LOSS_WEIGHTS))
        print(
            "Transfer learning model: {}".format(
                self.continue_model if self.continue_model else "None"
            )
        )
        print("=======================================")

    def _report_model(self, model, input, loader):
        print("=======================================")
        for width in self.widths:
            model.set_width(width)
            flops = FlopCountAnalysis(model, input)
            # Flops
            # Facebook Research
            # Parameters
            # Facebook Research

            # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
            print(sum(p.numel() for p in model.parameters() if p.requires_grad))
            # https://pypi.org/project/ptflops/
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(flops.total(), sum([x for x in parameter_count(model).values()]))
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("-----------------------------")
            print(
                get_model_complexity_info(
                    model, (3, 128, 128), print_per_layer_stat=False
                )
            )
            print("-----------------------------")
            print("*****************************")
            print(thop.profile(model, (input,)))
            print("*****************************")
            print("?????????????????????????????")
            print(pthflops.count_ops(model, input))
            print("?????????????????????????????")
            # print(flops.by_operator())
            # print(flops.by_module())
            # print(flops.by_module_and_operator())
            # print(flop_count_table(flops))
        print("=======================================")

    def _evaluate(
        self,
        metrics,
        loader,
        model,
        dataset_name,
        device,
        loss_function,
        image_pred=False,
        epoch=0,
    ):
        """
        Evaluate performance and add to metrics.
        """
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            for width in self.widths:
                model.set_width(width)
                y_pred = model.forward(X)

                name = "{}/{}".format(dataset_name, int(width * 100))
                metrics.add_loss(
                    loss_function(y_pred, y).cpu(),
                    name,
                )

                # This also works for cofly as lettuce rows just stay empty
                y_pred = get_binary_masks_infest(y_pred)
                metrics.add_jaccard(y, y_pred, name)
                metrics.add_precision(y, y_pred, name)
                if width == self.widths[-1] and image_pred:
                    metrics.add_image(X, y, y_pred, epoch)
        return metrics

    def _find_best_fitting(self, metrics):
        """
        Could you perhaps try training it by monitoring the validation scores for each
        width and then stopping the training at the epoch which maximises the difference
        between the widths when they are in the right order?

        Compare current metrics to best fitting and overwrite them if new best
        fitting were found given to a heuristic we have to come up with.

        Return True if best fitting was found, otherwise false.
        """
        metrics = [
            metrics["Jaccard/valid/25/weeds"],
            metrics["Jaccard/valid/50/weeds"],
            metrics["Jaccard/valid/75/weeds"],
            metrics["Jaccard/valid/100/weeds"],
        ]
        # print()
        # print(f"Comparing metrics: {metrics}")
        # print(f"Current best:      {self.best_fitting}")

        # First check if the widths are in order.
        for i, m in enumerate(metrics):
            if i == 0:
                continue
            if metrics[i - 1] > m:
                # print("Metrics not in order, returning false.")
                return False

        # Then check if the differences between neighbours are higher than current best
        if sum(
            [self.best_fitting[i] - self.best_fitting[i - 1] for i in range(1, 4)]
        ) > sum([metrics[i] - metrics[i - 1] for i in range(1, 4)]):
            return False

        self.best_fitting = metrics
        return True

    def _learning_rate_scheduler(self, optimizer):
        if self.learning_rate_scheduler == "no scheduler":
            return None
        elif self.learning_rate_scheduler == "linear":
            return LinearLR(
                optimizer,
                start_factor=1,
                end_factor=0,
                total_iters=self.epochs,
            )
        elif self.learning_rate_scheduler == "exponential":
            return ExponentialLR(
                optimizer,
                0.99,
            )

    def train(self):
        if self.verbose:
            print("Training process starting...")
            self._report_settings()
        # Prepare the data for training and validation
        ii = ImageImporter(
            self.dataset,
            validation=True,
            sample=self.sample,
            smaller=self.image_resolution,
        )
        train, validation = ii.get_dataset()
        if self.verbose:
            print("Number of training instances: {}".format(len(train)))
            print("Number of validation instances: {}".format(len(validation)))

        # Wandb report startup
        garage_path = ""
        if settings.WANDB:
            run = wandb.init(
                project="agriadapt",
                entity="colosal",
                group=self.wandb_group,
                config={
                    "Architecture": self.architecture,
                    "Batch Size": self.batch_size,
                    "Epochs": self.epochs,
                    "Learning Rate": self.learning_rate,
                    "Learning Rate Scheduler": self.learning_rate_scheduler,
                    "L2 Regularisation": self.regularisation_l2,
                    "Image Resolution": self.image_resolution,
                    "Train Samples": len(train),
                    "Validation Samples": len(validation),
                    "Dropout": settings.DROPOUT,
                    "Dataset": self.dataset,
                    "Loss Function Weights": settings.LOSS_WEIGHTS,
                    "Transfer learning": self.continue_model
                    if self.continue_model
                    else "None",
                },
            )
            wname = run.name.split("-")
            garage_path = "garage/runs/{} {} {}/".format(
                wname[2].zfill(4), wname[0], wname[1]
            )
            os.mkdir(garage_path)

        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(validation, batch_size=self.batch_size, shuffle=False)

        # Prepare a weighted loss function
        loss_function = torch.nn.CrossEntropyLoss(
            weight=tensor(settings.LOSS_WEIGHTS).to(self.device)
        )

        # Prepare the model
        out_channels = len(settings.LOSS_WEIGHTS)
        if not self.continue_model:
            if self.architecture == "slim":
                model = SlimUNet(out_channels)
            elif self.architecture == "squeeze":
                model = SlimSqueezeUNet(out_channels)
                if self.dataset == "cofly":
                    model = SlimSqueezeUNetCofly(out_channels)
                # model = SlimPrunedSqueezeUNet(in_channels, dropout=self.dropout)
            else:
                raise ValueError("Unknown model architecture.")
        else:
            model = torch.load(
                Path(settings.PROJECT_DIR)
                / "andraz/training/garage/"
                / self.continue_model
            )

        # summary(model, input_size=(in_channels, 128, 128))
        model.to(self.device)

        # Prepare the optimiser
        optimizer = Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.regularisation_l2,
        )
        scheduler = self._learning_rate_scheduler(optimizer)

        # Prepare the metrics tracker
        m_names = (
            [
                "{}/{}/{}".format(x, y, int(z * 100))
                for x in ["Loss"]
                for y in ["train", "valid"]
                for z in self.widths
            ]
            + [
                "{}/{}/{}/{}".format(x, y, int(z * 100), w)
                for x in ["Jaccard", "Precision"]
                for y in ["train", "valid"]
                for z in self.widths
                for w in ["back", "weeds", "lettuce"]
            ]
            + ["learning_rate"]
            + ["Image"]
        )
        metrics = Metricise(m_names)

        for epoch in range(self.epochs):
            s = datetime.now()

            model.train()
            for X, y in train_loader:
                # Move to GPU
                X, y = X.to(self.device), y.to(self.device)
                # Reset optimiser
                optimizer.zero_grad()
                # For all set widths
                for width in self.widths:
                    # Set the current width
                    model.set_width(width)
                    # Forward pass
                    outputs = model.forward(X)
                    # Calculate loss function
                    loss = loss_function(outputs, y)
                    # Backward pass
                    loss.backward()
                # Update weights
                optimizer.step()

            if self.learning_rate_scheduler == "no scheduler":
                metrics.add_learning_rate(self.learning_rate)
            else:
                metrics.add_learning_rate(scheduler.get_last_lr())
                scheduler.step()

            model.eval()
            with torch.no_grad():
                # Training evaluation
                metrics = self._evaluate(
                    metrics, train_loader, model, "train", self.device, loss_function
                )
                # Validation evaluation
                metrics = self._evaluate(
                    metrics,
                    valid_loader,
                    model,
                    "valid",
                    self.device,
                    loss_function,
                    image_pred=epoch % 50 == 0,
                    epoch=epoch,
                )

            res = metrics.report(wandb)
            # Only save the model if it is best fitting so far
            if self._find_best_fitting(res):
                torch.save(
                    model, garage_path + "model_{}.pt".format(str(epoch).zfill(4))
                )
            if self.verbose and epoch % 10 == 0:
                print(
                    "Epoch {} completed. Running time: {}".format(
                        epoch + 1, datetime.now() - s
                    )
                )

        if settings.WANDB:
            wandb.finish()
        torch.save(model, garage_path + "model_final.pt".format(epoch))


if __name__ == "__main__":
    # Train on GPU if available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # for sample_size in [10, 25, 50, 100]:
    # We need to train the new geok models of different sizes with and without transfer learning from cofly dataset
    # We do this for both sunet and ssunet
    # for architecture in ["slim", "squeeze"]:
    architecture = "squeeze"
    for image_resolution, batch_size in zip(
        [(128, 128), (256, 256), (512, 512)],
        [2**5, 2**3, 2**1],
    ):
        # tr = Training(
        #     device,
        #     dataset="geok",
        #     image_resolution=image_resolution,
        #     architecture=architecture,
        #     batch_size=batch_size,
        # )
        # tr.train()
        tr = Training(
            device,
            dataset="geok",
            image_resolution=image_resolution,
            architecture=architecture,
            batch_size=batch_size,
            continue_model="cofly_{}_{}.pt".format(architecture, image_resolution[0]),
        )
        tr.train()
