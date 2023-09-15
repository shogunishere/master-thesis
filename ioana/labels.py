from pathlib import Path

import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryPrecision

from andraz import settings
from andraz.data.data import ImageImporter
from KNN_model.KNN import KnnPrediction
from image_processing.spectral_features import SpectralFeatures
from image_processing.texture_features import TextureFeatures
from image_processing.vegetation_features import VegetationIndices

METRICS = {"precision": BinaryPrecision}

class Labels:
    def __init__(self, model, metrics=None, image_resolution=None):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_name = model.split(".")[0]
        self.model = self._load_model(model)
        self.model.eval()
        self.image_resolution = image_resolution
        self.image_resolution = image_resolution
        self.train, self.test = self._load_data()
        self.train_features, self.test_features = self._load_features()
        self.train_loader, self.test_loader = self._filter_data()
        self.metrics = METRICS if metrics is None else metrics

        self.results = {
            "train": {
                width: {
                    "precision": {"weeds": [], "back": []},
                } for width in settings.WIDTHS
            },
            "test": {
                width: {
                    "precision": {"weeds": [], "back": []},
                } for width in settings.WIDTHS
            },
        }

    @staticmethod
    def _load_model(model):
        return torch.load(Path(settings.PROJECT_DIR) / "/home/agriadapt/agriadapt/andraz/training/garage/" / model)
    
    def _load_data(self):
        ii = ImageImporter(
            "cofly",
            only_test=False,
            smaller=self.image_resolution,
        )
        train, test = ii.get_dataset()

        return train, test
    
    def _load_features(self):
        train_features = pd.read_pickle('/home/agriadapt/agriadapt/ioana/train_features.pickle')
        test_features = pd.read_pickle('/home/agriadapt/agriadapt/ioana/test_features.pickle')

        return train_features, test_features
    
    def _filter_data(self):
        train_features = pd.read_pickle('/home/agriadapt/agriadapt/ioana/train_features.pickle')
        test_features = pd.read_pickle('/home/agriadapt/agriadapt/ioana/test_features.pickle')
        train_indices = train_features["index"].tolist()
        test_indices = test_features["index"].tolist()
        filtered_train = []
        filtered_test = []

        for elem in range(len(self.train)):
            if elem in train_indices:
                sample = self.train[elem]
                filtered_train.append(sample)
        
        for elem in range(len(self.test)):
            if elem in test_indices:
                sample = self.test[elem]
                filtered_test.append(sample)

        train_data_loader = DataLoader(
            filtered_train, batch_size=1, shuffle=False
        )
        test_data_loader = DataLoader(
            filtered_test, batch_size=1, shuffle=False
        )

        return train_data_loader, test_data_loader
    
    def _infer(self, width):
        with torch.no_grad():
            self.model.set_width(width)

            for dataset_name, data_loader in [
                ("train", self.train_loader),
                ("test", self.test_loader),
            ]:
                for X, y in data_loader:
                    X = X.to("cuda:0")
                    y = y.to("cuda:0")
                    y_pred = self.model.forward(X)
                    y_pred = torch.where(y_pred < 0.5, 0, 1)

                    for metric in self.metrics:
                        for j, pred_class in enumerate(["weeds", "back"]):
                            self.results[dataset_name][width][metric][
                                pred_class
                            ].append(
                                self.metrics[metric](validate_args=False)
                                .to(self.device)(y[0][j], y_pred[0][j])
                                .cpu()
                            )

        return self.results["train"][width]["precision"]["weeds"], self.results["test"][width]["precision"]["weeds"]
    
    def _calculate_precision_mean(self, precision_scores, draw_graph=False):
        numeric_values = [value.item() for value in precision_scores]
        boxplot = plt.boxplot(numeric_values, vert=False)
        mean = "%.3f"% (boxplot['medians'][0].get_xdata()[0])

        if draw_graph:
            plt.figure()
            plt.title(f"Precision Score Distribution for 'weeds' {self.width} model {self.model_name}")
            plt.xlabel("Precision Score")
            plt.ylabel("Class: weeds")
            plt.show()

        return mean
    
    def _compute_labels(self, width, precision_list_train, precision_list_test, save_df=False):
        train_features = pd.read_pickle('/home/agriadapt/agriadapt/ioana/train_features.pickle')
        test_features = pd.read_pickle('/home/agriadapt/agriadapt/ioana/test_features.pickle')
        mean = float(self._calculate_precision_mean(precision_list_train))
        for features, precisions_list in [
                (train_features, precision_list_train),
                (test_features, precision_list_test),
            ]:
            label_column = f"label_{width}"
            index_list = features.index.tolist()
            for i in range(len(features)):
                precision_value = precisions_list[i]
                if precision_value.item() > mean:
                    features.at[index_list[i], label_column] = 1
                else:
                    features.at[index_list[i], label_column] = 0
        if save_df:
            train_path = f"/home/agriadapt/agriadapt/ioana/train_features_with_labels_width_{width}.pickle"
            train_features = train_features.to_pickle(train_path)
            print(f"Train data has been saved to path {train_path}")

            test_path = f"/home/agriadapt/agriadapt/ioana/test_features_with_labels_width_{width}.pickle"
            test_features = test_features.to_pickle(test_path)
            print(f"Test data has been saved to {test_path}")
        
        return train_features, test_features
    
    def run(self):
        # self._load_data()
        for width, no_neighbors in settings.KNN_WIDTHS.items():
            self.train_loader, self.test_loader = self._filter_data()
            precision_list_train, precision_list_test = self._infer(width)
            train_dataframe, test_dataframe = self._compute_labels(width, precision_list_train, precision_list_test)

            knn_model = KnnPrediction(width, train_dataframe, test_dataframe, no_neighbors)
            X_train, y_train, X_test, y_test = knn_model.load_data()
            X_train, X_test = knn_model.scale_data(X_train, X_test)
            y_pred = knn_model.fit_model(X_train, y_train, X_test, y_test)

            # finally, drop the lines that have label 1 in train_dataframe
            train_dataframe = train_dataframe.drop(train_dataframe[train_dataframe[f'label_{width}'] == 1].index)
            train_dataframe = train_dataframe.drop([f'label_{width}'], axis=1)

            test_dataframe = test_dataframe.drop(test_dataframe[test_dataframe[f'label_{width}'] == 1].index)
            test_dataframe = test_dataframe.drop([f'label_{width}'], axis=1)
            
            self.train_features = train_dataframe
            self.train_features.to_pickle(f"/home/agriadapt/agriadapt/ioana/train_features.pickle")
            self.test_features = test_dataframe
            self.test_features.to_pickle(f"/home/agriadapt/agriadapt/ioana/test_features.pickle")
        


if __name__ == "__main__":
    train_features, test_features = [], []
    ii = ImageImporter("cofly")
    train, test = ii.get_dataset()
    train_images = ii.tensor_to_image(train.X)
    test_images = ii.tensor_to_image(test.X)
    image_lists = [train_images, test_images]
    column_headers = ["index", "mean_brightness", "std_brightness", "max_brightness", "min_brightness",
                    "no_bins", "contrast_hue_hist", "std_hue_arc", "contrast", "mean_saturation",
                    "std_saturation", "max_saturation", "min_saturation", "keypoints", "ExG_ExR", 
                    "CIVE_index","glcm_contrast_1","glcm_contrast_2","glcm_contrast_3","glcm_contrast_4", 
                    "glcm_correlation_1","glcm_correlation_2", "glcm_correlation_3", "glcm_correlation_4",
                    "glcm_dissimilarity_1","glcm_dissimilarity_2", "glcm_dissimilarity_3","glcm_dissimilarity_4",
                    "glcm_asm_1", "glcm_asm_2", "glcm_asm_3", "glcm_asm_4",
                    "glcm_energy_1", "glcm_energy_2", "glcm_energy_3", "glcm_energy_4",
                    "glcm_homogeneity_1", "glcm_homogeneity_2", "glcm_homogeneity_3", "glcm_homogeneity_4"]

    for list_index, image_list in enumerate(image_lists):
        list_of_dictionaries = []
        for i, image in enumerate(image_list):
        
            spectral_features = SpectralFeatures(image)
            texture_features = TextureFeatures(image)
            vegetation_index = VegetationIndices(image)

            # color features
            mean_brightness, std_brightness, max_brightness, min_brightness = spectral_features.compute_brightness()
            no_bins, contrast_hue_hist, std_hue_arc = spectral_features.compute_hue_histogram()
            contrast = spectral_features.compute_contrast()
            mean_saturation, std_saturation, max_saturation, min_saturation = spectral_features.compute_saturation()
            keypoints = spectral_features.compute_sift_feats()

            # vegetation indices
            ExG = vegetation_index.excess_green_index()
            ExR = vegetation_index.excess_red_index()
            ExG_ExR_img = vegetation_index.excess_green_excess_red_index(ExG, ExR)
            CIVE_index = vegetation_index.colour_index_vegetation_extraction()
            binary_CIVE_image = vegetation_index.visualization_CIVE_Otsu_threshold(CIVE_index)

            # texture features
            glcm_matrix = texture_features.compute_glcm()
            glcm_contrast = np.ravel(texture_features.contrast_feature(glcm_matrix))
            glcm_dissimilarity = np.ravel(texture_features.dissimilarity_feature(glcm_matrix))
            glcm_homogeneity = np.ravel(texture_features.homogeneity_feature(glcm_matrix))
            glcm_energy = np.ravel(texture_features.energy_feature(glcm_matrix))
            glcm_correlation = np.ravel(texture_features.correlation_feature(glcm_matrix))
            glcm_asm = np.ravel(texture_features.asm_feature(glcm_matrix))
            
            features_dictionary = {
                "index" : i,
                "mean_brightness": mean_brightness,
                "std_brightness": std_brightness,
                "max_brightness": max_brightness,
                "min_brightness": min_brightness,
                "no_bins": no_bins,
                "contrast_hue_hist": contrast_hue_hist,
                "std_hue_arc": std_hue_arc,
                "contrast": contrast,
                "mean_saturation": mean_saturation,
                "std_saturation": std_saturation,
                "max_saturation": max_saturation,
                "min_saturation": min_saturation,
                "keypoints": keypoints,
                "ExG_ExR": ExG_ExR_img,
                "CIVE_index": float(binary_CIVE_image),
                "glcm_contrast_1": glcm_contrast[0],
                "glcm_contrast_2": glcm_contrast[1],
                "glcm_contrast_3": glcm_contrast[2],
                "glcm_contrast_4": glcm_contrast[3],
                "glcm_correlation_1": glcm_correlation[0],
                "glcm_correlation_2": glcm_correlation[1],
                "glcm_correlation_3": glcm_correlation[2],
                "glcm_correlation_4": glcm_correlation[3],
                "glcm_dissimilarity_1": glcm_dissimilarity[0],
                "glcm_dissimilarity_2": glcm_dissimilarity[1],
                "glcm_dissimilarity_3": glcm_dissimilarity[2],
                "glcm_dissimilarity_4": glcm_dissimilarity[3],
                "glcm_asm_1": glcm_asm[0],
                "glcm_asm_2": glcm_asm[1],
                "glcm_asm_3": glcm_asm[2],
                "glcm_asm_4": glcm_asm[3],
                "glcm_energy_1": glcm_energy[0],
                "glcm_energy_2": glcm_energy[1],
                "glcm_energy_3": glcm_energy[2],
                "glcm_energy_4": glcm_energy[3],
                "glcm_homogeneity_1": glcm_homogeneity[0],
                "glcm_homogeneity_2": glcm_homogeneity[1],
                "glcm_homogeneity_3": glcm_homogeneity[2],
                "glcm_homogeneity_4": glcm_homogeneity[3],
            }

            list_of_dictionaries.append(features_dictionary)
        
        if list_index == 0:
            train_features = pd.DataFrame(list_of_dictionaries, columns=column_headers, index=None)
        else:
            test_features = pd.DataFrame(list_of_dictionaries, columns=column_headers, index=None)

    file_name = '/home/agriadapt/agriadapt/ioana/train_features.pickle'
    train_features.to_pickle(file_name)
    print(f"Training features saved to {file_name}")

    file_name = '/home/agriadapt/agriadapt/ioana/test_features.pickle'
    test_features.to_pickle(file_name)
    print(f"Testing features saved to {file_name}")

    labels = Labels(
        "cofly_slim_256.pt"
    )
    labels.run()