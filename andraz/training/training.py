from datetime import datetime

import torch
import wandb as wandb
from torch import tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

import andraz.settings as settings
from andraz.data.data import ImageImporter
from andraz.helpers.masking import get_binary_masks_infest
from andraz.helpers.metricise import Metricise
from andraz.models.slim_unet import SlimUNet


class Training:
    def __init__(self):
        pass

    def _report_settings(self):
        print("=======================================")
        print("Training with the following parameters:")
        print("Epochs: {}".format(settings.EPOCHS))
        print("Learning rate: {}".format(settings.LEARNING_RATE))
        print("Batch size: {}".format(settings.BATCH_SIZE))
        print("Network widths: {}".format(settings.WIDTHS))
        print("=======================================")

    def train(self):
        print("Training process starting...")
        self._report_settings()

        # Wandb report startup
        if settings.WANDB:
            wandb.init(project="agriadapt", entity="colosal", config={})

        # Train on GPU if available
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        # Prepare the data for training and validation
        ii = ImageImporter("infest", validation=True, sample=True)
        train, validation = ii.get_dataset()
        train_loader = DataLoader(train, batch_size=settings.BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(
            validation, batch_size=settings.BATCH_SIZE, shuffle=False
        )

        # Prepare a weighted loss function
        loss_function = torch.nn.CrossEntropyLoss(
            # For cofly
            # weight=tensor([0.16, 0.28, 0, 0.28, 0.28]).to(device)
            # For infest
            weight=tensor([0.2, 0.4, 0.4]).to(device)
        )

        # Prepare the model
        in_channels = 3
        model = SlimUNet(in_channels)
        model.to(device)

        # Prepare the optimiser
        optimizer = Adam(model.parameters(), lr=settings.LEARNING_RATE)

        # Prepare the metrics tracker
        m_names = [
            "{}/{}/{}".format(x, y, int(z * 100))
            for x in ["Loss"]
            for y in ["train", "valid"]
            for z in settings.WIDTHS
        ] + [
            "{}/{}/{}/{}".format(x, y, int(z * 100), w)
            for x in ["Jaccard"]
            for y in ["train", "valid"]
            for z in settings.WIDTHS
            for w in ["back", "weeds", "lettuce"]
        ]
        metrics = Metricise(m_names)

        for epoch in range(settings.EPOCHS):
            s = datetime.now()
            model.train()

            for X, y in train_loader:
                # Move to GPU
                X, y = X.to(device), y.to(device)
                # Reset optimiser
                optimizer.zero_grad()
                # For all set widths
                for width in settings.WIDTHS:
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

            model.eval()
            with torch.no_grad():
                # Training evaluation
                for X, y in train_loader:
                    X, y = X.to(device), y.to(device)
                    for width in settings.WIDTHS:
                        model.set_width(width)
                        y_pred = model.forward(X)

                        name = "train/{}".format(int(width * 100))
                        metrics.add_loss(
                            loss_function(y_pred, y).cpu(),
                            name,
                        )

                        y_pred = get_binary_masks_infest(y_pred)
                        metrics.add_jaccard(y, y_pred, name)

                # Validation evaluation
                for X, y in valid_loader:
                    X, y = X.to(device), y.to(device)
                    for width in settings.WIDTHS:
                        model.set_width(width)
                        y_pred = model.forward(X)

                        name = "valid/{}".format(int(width * 100))
                        metrics.add_loss(
                            loss_function(y_pred, y).cpu(),
                            name,
                        )

                        y_pred = get_binary_masks_infest(y_pred)
                        metrics.add_jaccard(y, y_pred, name)
            metrics.report_wandb(wandb)

            if epoch % 100 == 0:
                torch.save(model, "slim_model_{}.pt".format(epoch))

            print(
                "Epoch {} completed. Running time: {}".format(epoch, datetime.now() - s)
            )

        if settings.WANDB:
            wandb.finish()
        torch.save(model, "slim_model_{}.pt".format(epoch))


if __name__ == "__main__":
    tr = Training()
    tr.train()
