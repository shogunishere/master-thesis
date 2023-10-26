# AgriAdapt

<img src="agriadapt.png" height="250">

An open source framework that performs on-UAV efficient weeds detection while flying over fields of crops. The backbone
of the system are [Slimmable Neural Networks](https://github.com/JiahuiYu/slimmable_networks) which enables our system
to maintain high level of performance while minimising the energy required to detect weeds. The repository is split into
two parts, one managing the image segmentation models, while the other is tasked with model width adaptation which is
the core of any slimmable neural network system.

Furthermore, the repository includes a novel dataset of 158 labelled images of lettuce fields, taken from a UAV.
Alongside the images, there are also the belonging segmentation masks that include labels of both, lettuce and weeds
classes.

## Reference

If you use this library or data in your work, please cite:

**Machidon, O. M., Krašovec, A., Machidon, A. L., Pejović, V., Latini, D., Sasidharan, S. T., & Del Frate, F. (2023,
October). AgriAdapt: Towards Resource-Efficient UAV Weed Detection using Adaptable Deep Learning. In Proceedings of the
2nd Workshop on Networked Sensing Systems for a Sustainable Society (pp. 193-200).**

## Machine Learning Pipeline

The machine learning pipeline consists of training logic for both, the segmentation models that are tasked to infer
position of weeds from a UAV image of a field, and the adaptation models that determine the width of the segmentation
model used on a per-image basis.

### Segmentation

The segmentation part of the library is tasked with defining, training, and saving of the weeds segmentation models. The
structure of the directory:

```
|-- segmentation
    |-- data (includes the collected dataset and preprocessing logic)
    |-- evaluation (currently not used)
    |-- helpers (various utility functions)
    |-- models (PyTorch definitions of model architecture)
    |-- training (model training logic)
        |-- garage (pretrained models)
```

To train a new segmentation model, define the model in the *models* directory and run the training script. Inspect
already implemented models for inspiration.

### Adaptation

The adaptation part of the library defines the adaptation models, their training and saving, and the image feature
extraction required to generate the models. Structure of the directory:

```
|-- adaptation
    |-- garage (pretrained adaptation models)
    |-- image_processing (image feature extraction)
    |-- KNN_model (KNN model training logic)
    |-- feature_selection.py (feature selection of extracted image features)
    |-- inference.py (get selected model widht for a specific image)
    |-- labels.py (generate KNN adaptation models for a specific image)
```

## Dataset

The dataset contains 158 images of a field of lettuce with various degree of weeds present in each image. The images are
split into a training (80%), validation (10%), and testing (10%) set and are located in the *segmentation/data/geok/*
directory. Also included are the segmentation masks of both, weeds and lettuce classes, formatted in YOLOv7 format. All
logic required to obtain the images, including the segmentation masks can be found in the data.py in
*segmentation/data*.

## Inference Scripts

To investigate the performance of different segmentation and adaptation approaches, there are also two demo scripts
present in the *demo/* root folder of the repository.

**run_model_inference.py** calculates different metrics (accuracy, precision, recall, f1 score, Jaccard index) for a
predefined set of trained models. It also calculates the oracle predictor, which selects the best performing width of a
model for every image and represents a ceiling result in terms of width selection for a given segmentation model.

**run_single_inference.py** performs weeds segmentation of a single image for a single given model. This script can be
used as a starting point for on-device inference.

## License

This work is licensed
under [Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1). 