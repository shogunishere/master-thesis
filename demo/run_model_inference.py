# In short, what we need is a regular U-Net and squeeze U-Net trained on the CoFly-WeedDB dataset
# (using the methodology from the paper, with all weed classes grouped together in a single “Weed”
# label). Ideally, for both networks we should have a basic test bench script that would load the
# network, perform inference on the test set, and report both IoU and precision.
import os
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from plotly import graph_objects as go
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryJaccardIndex,
)
from torchvision.transforms import transforms

from andraz import settings
from andraz.data.data import ImageImporter
from andraz.helpers.drive_fetch import setup_env
from andraz.helpers.masking import get_binary_masks_infest
from andraz.helpers.metricise import Metricise
from ioana.inference import AdaptiveWidth


class Inference:
    def __init__(
        self,
        model,
        metrics=None,
        image_resolution=None,
        create_images=False,
        dataset="cofly",
    ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_name = model.split(".")[0]
        self.create_images = create_images
        self.model = self._load_test_model(model)
        self.model.eval()
        self.image_resolution = image_resolution
        self.metrics = settings.METRICS if metrics is None else metrics
        self.results = {
            x: {
                "iou": {"back": [], "weeds": []},
                "precision": {"back": [], "weeds": []},
                "recall": {"back": [], "weeds": []},
                "f1score": {"back": [], "weeds": []},
            }
            for x in settings.WIDTHS + ["adapt"]
        }
        self.width_distribution = {x: 0 for x in settings.WIDTHS}
        # self.width_selection = AdaptiveWidth(
        #     garage_dir=Path(settings.PROJECT_DIR) / "ioana/garage" / self.model_name
        # )
        self.width_selection_widths = []
        self.tensor_to_image = ImageImporter("cofly").tensor_to_image
        self.dataset = dataset

    @staticmethod
    def _load_test_model(model):
        return torch.load(
            Path(settings.PROJECT_DIR) / "andraz/training/garage/" / model
        )

    def _load_test_data(self):
        ii = ImageImporter(
            self.dataset,
            # validation=True,
            smaller=self.image_resolution,
            only_test=True,
        )
        _, test = ii.get_dataset()
        return DataLoader(test, batch_size=1, shuffle=False)

    def _infer(self):
        test_loader = self._load_test_data()
        # Prepare the metrics tracker
        m_names = (
            [
                "{}/{}/{}/{}".format(x, y, int(z * 100), w)
                for x in ["Jaccard", "Precision"]
                for y in ["train", "valid"]
                for z in settings.WIDTHS
                for w in ["back", "weeds"]
            ]
            + ["learning_rate"]
            + ["Image"]
        )
        metrics = Metricise(m_names)
        # Infer for all available widths of the model
        with torch.no_grad():

            for width in settings.WIDTHS:
                self.model.set_width(width)
                metrics = self._evaluate(
                    metrics, test_loader, self.model, "valid", "cuda:0", None
                )
            self.results = metrics.report(None)
            # i = 1
            # for X, y in test_loader:
            #     # Resize the image to the required input size
            #     X = X.to("cuda:0")
            #     y = y.to("cuda:0")
            #     y_pred = self.model.forward(X)
            #     y_pred = torch.where(y_pred < 0.5, 0, 1)
            #
            #     if self.create_images:
            #         self._generate_mask_image(
            #             X[0].permute(1, 2, 0).cpu(),
            #             y[0][1].cpu(),
            #             y_pred[0][1].cpu(),
            #             int(width * 100),
            #             i,
            #         )
            #     metrics
            #     for metric in self.metrics:
            #         for j, pred_class in enumerate(["back", "weeds"]):
            #             self.results[width][metric][pred_class].append(
            #                 self.metrics[metric](validate_args=False)
            #                 .to(self.device)(y[0][j], y_pred[0][j])
            #                 .cpu()
            #             )
            #     i += 1
        # # Infer with the adaptation algorithm
        # with torch.no_grad():
        #     # We load the test loader again, as we need full-size
        #     # images in order for the width selection to work properly
        #     i = 1
        #     for X, y in test_loader:
        #         # Get the image width and set the model to it
        #         image = self.tensor_to_image(X)[0]
        #         width = self.width_selection.get_image_width(image)
        #         self.width_selection_widths.append(width)
        #         self.model.set_width(width)
        #         self.width_distribution[width] += 1
        #         # Resize the image to the required input size
        #         X = X.to("cuda:0")
        #         y = y.to("cuda:0")
        #         y_pred = self.model.forward(X)
        #         y_pred = torch.where(y_pred < 0.5, 0, 1)
        #
        #         if self.create_images:
        #             self._generate_mask_image(
        #                 X[0].permute(1, 2, 0).cpu(),
        #                 y[0][1].cpu(),
        #                 y_pred[0][1].cpu(),
        #                 "adapt",
        #                 i,
        #             )
        #
        #         for metric in self.metrics:
        #             for j, pred_class in enumerate(["back", "weeds"]):
        #                 self.results["adapt"][metric][pred_class].append(
        #                     self.metrics[metric](validate_args=False)
        #                     .to(self.device)(y[0][j], y_pred[0][j])
        #                     .cpu()
        #                 )
        # self._average_results()
        return self.results

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
            for width in settings.WIDTHS:
                model.set_width(width)
                y_pred = model.forward(X)

                name = "{}/{}".format(dataset_name, int(width * 100))
                # This also works for cofly as lettuce rows just stay empty
                y_pred = get_binary_masks_infest(y_pred)
                metrics.add_jaccard(y, y_pred, name)
                metrics.add_precision(y, y_pred, name)
        return metrics

    def _average_results(self):
        """
        Average results per-width, for adaptation algorithm, and calculate
        the oracle prediction -> optimal results of predictions of different widths.
        """
        oracle = {
            "iou": {"back": [], "weeds": [], "backwidths": [], "weedswidths": []},
            "precision": {"back": [], "weeds": [], "backwidths": [], "weedswidths": []},
            "recall": {"back": [], "weeds": [], "backwidths": [], "weedswidths": []},
            "f1score": {"back": [], "weeds": [], "backwidths": [], "weedswidths": []},
        }
        for width_mult in settings.WIDTHS + ["adapt"]:
            for metric in self.metrics:
                for _, pred_class in enumerate(["back", "weeds"]):
                    # Calculate for oracle
                    if width_mult == 0.25:
                        oracle[metric][pred_class] = self.results[width_mult][metric][
                            pred_class
                        ]
                        oracle[metric][pred_class + "widths"] = np.repeat(
                            width_mult, len(oracle[metric][pred_class])
                        )
                    elif width_mult in settings.WIDTHS:
                        for i, (x, o) in enumerate(
                            zip(
                                self.results[width_mult][metric][pred_class],
                                oracle[metric][pred_class],
                            )
                        ):
                            oracle[metric][pred_class][i] = max(x, o)
                            if x > o:
                                oracle[metric][pred_class + "widths"][i] = width_mult
                    else:
                        pass
                    # Average results
                    self.results[width_mult][metric][pred_class] = np.mean(
                        self.results[width_mult][metric][pred_class]
                    )
        # Average the oracle results
        for metric in self.metrics:
            for _, pred_class in enumerate(["back", "weeds"]):
                oracle[metric][pred_class] = np.mean(oracle[metric][pred_class])
                oracle[metric][pred_class + "widths"] = np.mean(
                    oracle[metric][pred_class + "widths"]
                )
        self.results["adapt"]["avgwidth"] = np.mean(self.width_selection_widths)
        self.results["oracle"] = oracle

    def _generate_mask_image(self, x, y, pred, width, i):
        save_path = "images/{}".format(self.model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if width == 25:
            plt.imshow(x)
            plt.savefig("{}/{}_{}.png".format(save_path, str(i).zfill(4), "image"))
            plt.imshow(y)
            plt.savefig("{}/{}_{}.png".format(save_path, str(i).zfill(4), "mask"))
        plt.imshow(pred)
        plt.savefig("{}/{}_{}_{}.png".format(save_path, str(i).zfill(4), "pred", width))

    def run(self):
        self._infer()


class Comparator:
    def __init__(self, models, metrics=None, graphs=False):
        self.models = models
        self.metrics = settings.METRICS if metrics is None else metrics
        self.graphs = graphs

    def _draw_tab(self, results):
        for pred_class in ["back", "weeds"]:
            for metric in ["Jaccard", "Precision"]:
                print(
                    "======================================================================================"
                )
                print(metric, pred_class)
                print()
                print("{:10s}".format(""), end="")
                for model, _ in self.models:
                    print("{:20s}".format(model), end=" ")
                print()
                for width in settings.WIDTHS:
                    print("{:9s}".format(str(width)), end=" ")
                    for model, _ in self.models:
                        print(
                            "{:20s}".format(
                                str(
                                    round(
                                        results[model][
                                            f"{metric}/valid/{str(int(width * 100))}/{pred_class}"
                                        ]
                                        * 100,
                                        2,
                                    )
                                )
                                + " %"
                            ),
                            end=" ",
                        )
                    print()

                # # Print average width selected for adaptation alg
                # print("{:9s}".format("adapt w"), end=" ")
                # for model, _ in self.models:
                #     print(
                #         "{:20s}".format(
                #             str(
                #                 round(
                #                     results[model]["adapt"]["avgwidth"],
                #                     2,
                #                 )
                #             )
                #         ),
                #         end=" ",
                #     )
                # print()

                # Print average width selected for oracle
                # print("{:9s}".format("oracle w"), end=" ")
                # for model, _ in self.models:
                #     print(
                #         "{:20s}".format(
                #             str(
                #                 round(
                #                     results[model]["oracle"][metric][
                #                         pred_class + "widths"
                #                     ],
                #                     2,
                #                 )
                #             )
                #         ),
                #         end=" ",
                #     )
                # print()

                print()

    def _draw_graph(self, model, results, mean_width):
        for pred_class in ["weeds", "back"]:
            for metric in self.metrics:
                metric_scores = []
                for width in settings.WIDTHS + ["adapt"]:
                    score = round(results[width][metric][pred_class] * 100, 2)
                    metric_scores.append(score)
                x = settings.WIDTHS + ["adaptive"]
                y = metric_scores
                x_ticks = settings.WIDTHS + [round(mean_width, 2)]

                plt.plot(
                    x_ticks[:-1], y[:-1], marker="o", linestyle="-", label="widths"
                )
                plt.plot(
                    [round(mean_width, 2)],
                    [y[-1]],
                    marker="o",
                    color="red",
                    label="adaptive",
                )
                plt.xlabel("widths")
                plt.xticks(x_ticks, x_ticks)
                plt.ylabel(f"{metric} scores")
                plt.title(f"{metric} {pred_class} scores for model {model}")
                plt.legend()
                plt.show()

    def run(self):
        # Select a model from andraz/training/garage directory and set the
        # image resolution tuple to match the image input size of the model
        results = {}
        graph = False
        for model, size in self.models:
            infer = Inference(
                model,
                image_resolution=(size, size),
                create_images=False,
                dataset="geok",
            )
            infer.run()
            results[model] = infer.results

            if graph:
                x = [k for k in infer.width_distribution]
                y = [v for _, v in infer.width_distribution.items()]
                fig = go.Figure(
                    data=go.Bar(
                        x=x,
                        y=y,
                    )
                )
                fig.update_layout(
                    {"title": "Distribution of selected widths for test images."}
                )
                fig.update_xaxes(type="category")
                fig.show()
                # Only show once as all the plots will be the same
                graph = False

            weighted_sum = sum(
                key * value for key, value in infer.width_distribution.items()
            )
            total_weight = sum(infer.width_distribution.values())
            # weighted_mean = weighted_sum / total_weight

            if self.graphs:
                self._draw_graph(model, results[model], 1)

        self._draw_tab(results)


if __name__ == "__main__":
    # Download the geok dataset and place it in a proper directory.
    # You only have to do this the first time, afterwards the data is ready to go.
    # setup_env()

    models = [
        ("geok_squeeze_128_trans_opt.pt", 128),
        ("geok_squeeze_256_trans_opt.pt", 256),
        ("geok_squeeze_512_trans_opt.pt", 512),
    ]
    comparator = Comparator(models)

    # Run inference for multiple models and display comparative tables
    comparator.run()
