from datetime import datetime
from time import sleep

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import unique
from torch import argmax, tensor, cat
from torch.cuda import memory_summary, mem_get_info
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.utils import save_image, draw_segmentation_masks
from plotly import graph_objects as go

import andraz.settings as settings
from andraz.data.data import ImageImporter


class EvaluationHelper:
    def __init__(
        self,
        device="cuda:0",
        dataset="cofly",
        visualise=False,
        class_num=5,
    ):
        self.device = device
        self.dataset = dataset
        self.visualise = visualise
        self.class_num = class_num

        self.test_loader = None

    def import_data(self):
        """
        Import dataset that is used for evaluation.
        """
        ii = ImageImporter(self.dataset)
        train, test = ii.get_dataset()
        self.test_loader = DataLoader(test, batch_size=1, shuffle=False)

    def evaluate(self, model_path):
        """
        Evaluate a given method with requested metrics.
        Optionally also visualise segmentation masks/overlays.
        """
        if not self.test_loader:
            print("Import data first.")
            return
        self.model = torch.load(model_path)
        self.model.eval()
        # https://deci.ai/blog/measure-inference-time-deep-neural-networks/
        starter, ender = starter, ender = torch.cuda.Event(
            enable_timing=True
        ), torch.cuda.Event(enable_timing=True)

        i = 0
        jac_scores = {x: [] for x in settings.width_mult_list}
        run_times = {x: [] for x in settings.width_mult_list}
        with torch.no_grad():
            for X, y in self.test_loader:
                first = True
                X = X.to(device)
                torch.cuda.empty_cache()
                w = 1
                # Warmup to have a fair running time measurements
                self.model.forward(X)
                for width_mult in settings.width_mult_list:
                    self.model.set_width(width_mult)
                    starter.record()
                    images = self.model.forward(X)
                    ender.record()
                    torch.cuda.synchronize()
                    run_times[width_mult].append(starter.elapsed_time(ender))

                    # Save segmentation masks
                    if self.visualise:
                        self._save_images(X, y, images, i, w, first)

                    # Calculate Jaccard Index
                    y_jac = argmax(y[0], dim=0).to(device)
                    image = argmax(images[0], dim=0).to(device)
                    jac_scores[width_mult].append(
                        self.evaluate_jaccard(y_jac, image, self.class_num)
                    )
                    first = False

                    w += 1
                i += 1

        return jac_scores, run_times

    def evaluate_jaccard(self, y, pred, class_num):
        """
        Use the Jaccard Index to evaluate model's performance.
        """
        jaccard = MulticlassJaccardIndex(class_num).to(self.device)
        return jaccard(y, pred).cpu()

    def _save_images(self, X, y, pred, batch, width, first):
        for i, (x, test, y) in enumerate(zip(X, y, pred)):
            # Generate an original rgb image with predicted mask overlay.
            x_mask = torch.tensor(
                torch.mul(x.clone().detach().cpu(), 255), dtype=torch.uint8
            )
            mask = argmax(y.clone().detach(), dim=0)
            # print(mask)
            # print(mask.shape)
            # print(unique(mask.cpu()))
            weed_mask = torch.where(mask == 1, True, False)[None, :, :]
            lettuce_mask = torch.where(mask == 2, True, False)[None, :, :]
            mask = cat((weed_mask, lettuce_mask), 0)
            # print(weed_mask.shape)
            # print(lettuce_mask.shape)
            # print(x_mask.shape)
            # print(mask.shape)
            # print(unique(mask.cpu()))
            image = draw_segmentation_masks(
                x_mask, mask, colors=["red", "green"], alpha=0.5
            )
            plt.imshow(image.permute(1, 2, 0))
            # plt.show()
            plt.savefig(
                "plots/infest/slim_unet/{}_{}_{}_overlay.png".format(batch, i, width)
            )

            # x = x.cpu().float().permute(1, 2, 0)
            # test = argmax(test, dim=0).float()
            # y = argmax(y, dim=0).cpu().float()
            # Generate separate images
            # plt.imshow(x)
            # plt.savefig("plots/{}_{}_org.png".format(batch, i))
            # if first:
            #     plt.imshow(test)
            #     plt.savefig("plots/slim_net/{}_{}_0_mask.png".format(batch, i))
            # plt.imshow(y)
            # plt.savefig("plots/{}_{}_pred.png".format(batch, i))

    @staticmethod
    def _report_memory():
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("Total: {}".format(t / 1024 / 1024))
        print("Reserved: {}".format(r / 1024 / 1024))
        print("Allocated: {}".format(a / 1024 / 1024))
        print()


if __name__ == "__main__":
    # Generate images
    # device = "cuda:0"
    # eh = EvaluationHelper(device=device, dataset="infest", class_num=3, visualise=True)
    # eh.import_data()
    #
    # model = "garage/slim_model_1000.pt"
    # jac_scores, run_times = eh.evaluate(model)

    # Generate plots
    device = "cuda:0"
    eh = EvaluationHelper(device=device, dataset="infest", class_num=3)
    eh.import_data()

    steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    jaccards = {x: [] for x in settings.width_mult_list}
    times = {x: [] for x in settings.width_mult_list}
    jac_std = {x: [] for x in settings.width_mult_list}
    tim_std = {x: [] for x in settings.width_mult_list}
    for step in steps:
        model = "garage/slim_model_{}.pt".format(step)

        jac_scores, run_times = eh.evaluate(model)
        for x in settings.width_mult_list:
            jaccards[x].append(np.mean(jac_scores[x]))
            times[x].append(np.mean(run_times[x]))
            jac_std[x].append(np.std(jac_scores[x]))
            tim_std[x].append(np.std(run_times[x]))

    jac_data = []
    tim_data = []
    for x in settings.width_mult_list:
        jac_data.append(
            go.Bar(
                x=steps,
                y=jaccards[x],
                name=str(x),
                error_y={"array": jac_std[x], "visible": True},
            )
        )
        tim_data.append(
            go.Bar(
                x=steps,
                y=times[x],
                name=str(x),
                error_y={"array": tim_std[x], "visible": True},
            )
        )
    fig = go.Figure(data=jac_data)
    fig.update_layout(
        {
            "title": "Jaccard index for different time steps and network widths",
            "xaxis_title": "Epoch",
            "yaxis_title": "Jaccard Index",
        }
    )
    fig.show()
    fig = go.Figure(data=tim_data)
    fig.update_layout(
        {
            "title": "Inference times for different time steps and network widths",
            "xaxis_title": "Epoch",
            "yaxis_title": "Inference time",
        }
    )
    fig.show()
