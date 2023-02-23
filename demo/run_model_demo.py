from pathlib import Path

import torch
from torch.utils.data import DataLoader

from andraz import settings
from andraz.data.data import ImageImporter
from andraz.helpers.drive_fetch import setup_env

# Before running the script create a local_settings.py
# file in the andraz directory then set the PROJECT_DIR
# variable to point to the project's root directory.

# Prepare the env
setup_env()

# Load the model
model = torch.load(
    Path(settings.PROJECT_DIR) / "training/garage/infest/slim_model_1000.pt"
)

# Load the dataset ("infest" -> labelled part of the dataset from Geo-K)
ii = ImageImporter("infest", sample=True)
train, test = ii.get_dataset()
test_loader = DataLoader(test, batch_size=1, shuffle=False)

for X, y in test_loader:
    X = X.to("cuda:0")

    # Iterate through different widths of the network
    for width_mult in settings.width_mult_list:
        model.set_width(width_mult)
        mask_pred = model.forward(X)
        print(mask_pred.shape)
