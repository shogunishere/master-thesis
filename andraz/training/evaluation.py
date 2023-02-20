from datetime import datetime
from time import sleep

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import argmax, tensor
from torch.cuda import memory_summary, mem_get_info
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.utils import save_image, draw_segmentation_masks

import andraz.settings as settings
from andraz.data.data import ImageImporter


class EvaluationHelper:
    def __init__(
        self,
        model_path="garage/model.pt",
        device="cuda:0",
        dataset="cofly",
        visualise=False,
    ):
        self.model = torch.load(model_path)
        self.device = device
        self.dataset = dataset
        self.visualise = visualise

    def evaluate(self):
        """
        Evaluate a given method with requested metrics.
        Optionally also visualise segmentation masks/overlays.
        """
        self.model.eval()
        ii = ImageImporter(self.dataset)
        train, test = ii.get_dataset()
        test_loader = DataLoader(test, batch_size=1, shuffle=False)

        i = 0
        jac_scores = {x: [] for x in settings.width_mult_list}
        with torch.no_grad():
            for X, y in test_loader:
                first = True
                X = X.to(device)
                torch.cuda.empty_cache()
                w = 1
                for width_mult in settings.width_mult_list:
                    self.model.set_width(width_mult)
                    images = self.model.forward(X)

                    # Save segmentation masks
                    if self.visualise:
                        self._save_images(X, y, images, i, w, first)

                    # Calculate Jaccard Index
                    y_jac = argmax(y[0], dim=0).to(device)
                    image = argmax(images[0], dim=0).to(device)
                    jac_scores[width_mult].append(self._evaluate_jaccard(y_jac, image))
                    first = False

                    w += 1
                i += 1

        for x in settings.width_mult_list:
            print("{}: {}".format(x, np.mean(jac_scores[x])))

    def _evaluate_jaccard(self, y, pred):
        """
        Use the Jaccard Index to evaluate model's performance.
        """
        jaccard = MulticlassJaccardIndex(5).to(self.device)
        return jaccard(y, pred).cpu()

    def _save_images(self, X, y, pred, batch, width, first):
        for i, (x, test, y) in enumerate(zip(X, y, pred)):
            # Generate an original rgb image with predicted mask overlay.
            x_mask = torch.tensor(
                torch.mul(x.clone().detach().cpu(), 255), dtype=torch.uint8
            )
            mask = torch.tensor(argmax(y.clone().detach(), dim=0), dtype=torch.bool)
            image = draw_segmentation_masks(x_mask, mask, colors="yellow", alpha=0.5)
            plt.imshow(image.permute(1, 2, 0))
            plt.show()
            # plt.savefig("plots/slim_net/{}_{}_{}_overlay.png".format(batch, i, width))

            x = x.cpu().float().permute(1, 2, 0)
            test = argmax(test, dim=0).float()
            y = argmax(y, dim=0).cpu().float()

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
    for model in [
        "garage/slim_model_0.pt",
        "garage/slim_model_100.pt",
        "garage/slim_model_200.pt",
        "garage/slim_model_300.pt",
    ]:
        print(model)
        device = "cuda:0"
        eh = EvaluationHelper(model_path=model, device=device)
        eh.evaluate()
        print()
