import os
from zipfile import ZipFile
from pathlib import Path

import gdown

from andraz import settings


def setup_env():
    project_dir = Path(settings.PROJECT_DIR)
    # Download the data
    setup_folder(
        "https://drive.google.com/file/d/1w2jbor9QR3iUQ0V1pphsmNHw4ZZdgf_f/view?usp=share_link",
        project_dir
        / "data/agriadapt/NN_labeled_samples_salad_infesting_plants.v1i.yolov7pytorch/",
    )
    # Download the models
    setup_folder(
        "https://drive.google.com/file/d/1F8wtjxjESxWRomzAxuR43tJ2bOptTMfT/view?usp=share_link",
        project_dir / "training/garage/",
    )


def setup_folder(url, path):
    # Check if path directory exists and create it if not
    if not os.path.exists(path):
        os.makedirs(path)
    # Download the zip file
    gdown.download(url, str(path / "tmp.zip"), quiet=False, fuzzy=True)
    # Unzip the downloaded file
    with ZipFile(path / "tmp.zip", "r") as zip_ref:
        zip_ref.extractall(path)
    # Remove the zip file
    os.remove(path / "tmp.zip")