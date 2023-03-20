from pathlib import Path

import numpy as np
import torch
from numpy import unique
from sklearn.metrics import precision_score
from torch.utils.data import DataLoader

from andraz import settings
from andraz.data.data import ImageImporter
from andraz.helpers.drive_fetch import setup_env
from andraz.helpers.masking import get_binary_masks_infest

CLASSES = ["back", "weeds", "lettuce"]


if __name__ == "__main__":
    # Before running the script create a local_settings.py
    # file in the andraz directory then set the PROJECT_DIR
    # variable to point to the project's root directory.

    # Prepare the env -- download data and models (only needed on data/models update)
    # Comment the line when not needed
    setup_env()

    # Load the model
    model = torch.load(
        Path(settings.PROJECT_DIR)
        / "training/garage/no_decay.pt"
        # / "training/garage/linear.pt"
        # / "training/garage/exponential.pt"
    )

    # Load the dataset ("infest" -> labelled part of the dataset from Geo-K)
    ii = ImageImporter("infest", only_test=True)
    _, test = ii.get_dataset()
    test_loader = DataLoader(test, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        # Iterate through different widths of the network
        for width_mult in settings.WIDTHS:
            print("Evaluating for width: {}".format(width_mult))
            model.set_width(width_mult)
            scores = {x: [] for x in CLASSES}

            for X, y in test_loader:
                X = X.to("cuda:0")
                mask_pred = model.forward(X)

                # Calculate precision for all classes and print them
                # Iterate through classes
                for j in range(y.shape[1]):
                    # If everything is predicted as negative
                    if len(unique(mask_pred[0][j].cpu())) == 1:
                        scores[CLASSES[j]].append(0)
                    else:
                        mask_pred = get_binary_masks_infest(mask_pred)
                        scores[CLASSES[j]].append(
                            # For the binary precision evaluation to work, we need 1D array
                            precision_score(
                                [
                                    int(xx)
                                    for x in y[0][j].cpu().numpy().tolist()
                                    for xx in x
                                ],
                                [
                                    int(xx)
                                    for x in mask_pred[0][j].cpu().numpy().tolist()
                                    for xx in x
                                ],
                            )
                        )
            for key in scores:
                print("{}: {}".format(key, round(np.mean(scores[key]), 3)))
            print()
