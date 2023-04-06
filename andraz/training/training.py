import os
from datetime import datetime

import numpy as np
import torch
import wandb as wandb
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
from torch import tensor, argmax
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, ExponentialLR
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info
import thop
import pthflops
from plotly import graph_objects as go

import andraz.settings as settings
from andraz.data.data import ImageImporter
from andraz.helpers.masking import get_binary_masks_infest
from andraz.helpers.metricise import Metricise
from andraz.helpers.model_profiling import model_profiling
from andraz.models.slim_unet import SlimUNet


class Training:
    def __init__(
        self,
        device,
        epochs=settings.EPOCHS,
        learning_rate=settings.LEARNING_RATE,
        learning_rate_scheduler=settings.LEARNING_RATE_SCHEDULER,
        batch_size=settings.BATCH_SIZE,
        regularisation_l2=settings.REGULARISATION_L2,
        image_resolution=settings.IMAGE_RESOLUTION,
        widths=settings.WIDTHS,
        verbose=1,
        wandb_group=None,
    ):
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.learning_rate_scheduler = learning_rate_scheduler
        self.batch_size = batch_size
        self.regularisation_l2 = regularisation_l2
        self.image_resolution = image_resolution
        self.widths = widths
        self.verbose = verbose
        self.wandb_group = wandb_group

    def _report_settings(self):
        print("=======================================")
        print("Training with the following parameters:")
        print("Epochs: {}".format(self.epochs))
        print("Learning rate: {}".format(self.learning_rate))
        print("Learning rate scheduler: {}".format(self.learning_rate_scheduler))
        print("Batch size: {}".format(self.batch_size))
        print("L2 regularisation: {}".format(self.regularisation_l2))
        print("Image resolution: {}".format(self.image_resolution))
        print("Network widths: {}".format(self.widths))
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

    def _evaluate(self, metrics, loader, model, dataset_name, device, loss_function):
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

                y_pred = get_binary_masks_infest(y_pred)
                metrics.add_jaccard(y, y_pred, name)
                metrics.add_precision(y, y_pred, name)

        return metrics

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
            "infest", validation=True, sample=True, smaller=self.image_resolution
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
                    "Batch Size": self.batch_size,
                    "Epochs": self.epochs,
                    "Learning Rate": self.learning_rate,
                    "Learning Rate Scheduler": self.learning_rate_scheduler,
                    "L2 Regularisation": self.regularisation_l2,
                    "Image Resolution": self.image_resolution,
                    "Train Samples": len(train),
                    "Validation Samples": len(validation),
                },
            )
            wname = run.name.split("-")
            garage_path = "garage/infest/{} {} {}/".format(
                wname[2].zfill(4), wname[0], wname[1]
            )
            os.mkdir(garage_path)

        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(validation, batch_size=self.batch_size, shuffle=False)

        # Prepare a weighted loss function
        loss_function = torch.nn.CrossEntropyLoss(
            # For cofly
            # weight=tensor([0.16, 0.28, 0, 0.28, 0.28]).to(self.device)
            # For infest
            weight=tensor([0.2, 0.4, 0.4]).to(self.device)
        )

        # Prepare the model
        in_channels = 3
        model = SlimUNet(in_channels)
        model.to(self.device)
        X, _ = next(iter(train_loader))
        X = X.to(self.device)

        # Reporting from slimmable networks
        for width in settings.WIDTHS:
            # Report on flops/parameters of the model
            print()
            model.set_width(width)
            model_profiling(model, X)
            print()

        # All other rando packages found online
        # self._report_model(model, X, train_loader)
        0 / 0

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
            # + ["Image"]
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
                    metrics, valid_loader, model, "valid", self.device, loss_function
                )

            metrics.report_wandb(wandb)
            torch.save(
                model, garage_path + "slim_model_{}.pt".format(str(epoch).zfill(4))
            )
            if self.verbose:
                print(
                    "Epoch {} completed. Running time: {}".format(
                        epoch, datetime.now() - s
                    )
                )

        if settings.WANDB:
            wandb.finish()
        torch.save(model, garage_path + "slim_model.pt".format(epoch))


if __name__ == "__main__":
    # Train on GPU if available
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    tr = Training(device)
    tr.train()
