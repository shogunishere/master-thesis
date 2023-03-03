import os

import torch
from matplotlib import pyplot as plt, animation
from torch import argmax, cat
from torch.utils.data import DataLoader
from torchvision.utils import draw_segmentation_masks

from andraz.data.data import ImageImporter


class GifFactory:
    def generate_images(self):
        model_dir = "../training/garage/infest/0002/"
        device = "cuda:0"

        ii = ImageImporter("infest", sample=True)
        train, test = ii.get_dataset()
        test_loader = DataLoader(test)

        for model_path in os.listdir(model_dir):
            model = torch.load(model_dir + model_path)
            model = model.to(device)
            model.eval()
            model.set_width(1)
            with torch.no_grad():
                it = iter(test_loader)
                next(it)
                X, y = next(it)
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                x_mask = torch.tensor(
                    torch.mul(X[0].clone().detach().cpu(), 255), dtype=torch.uint8
                )

                # Ground truth mask
                # mask = argmax(y[0].clone().detach(), dim=0)
                # Predicted mask
                mask = argmax(y_pred[0].clone().detach(), dim=0)
                weed_mask = torch.where(mask == 1, True, False)[None, :, :]
                lettuce_mask = torch.where(mask == 2, True, False)[None, :, :]
                mask = cat((weed_mask, lettuce_mask), 0)

                image = draw_segmentation_masks(
                    x_mask, mask, colors=["red", "green"], alpha=0.5
                )
                plt.title(model_path[:-3].split("_")[2].zfill(4))
                plt.imshow(image.permute(1, 2, 0))
                # plt.show()
                plt.savefig(
                    "plots/infest/gif/{}.{}".format(
                        model_path[:-3].split("_")[2].zfill(4), "png"
                    )
                )

    def generate_gif(self):
        # Create new figure for GIF
        fig, ax = plt.subplots()

        # Adjust figure so GIF does not have extra whitespace
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax.axis("off")
        ims = []

        image_dir = "plots/infest/gif/"
        for image in sorted(os.listdir(image_dir)):
            im = ax.imshow(plt.imread(image_dir + image), animated=True)
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=100)
        ani.save("masks.gif")


if __name__ == "__main__":
    gf = GifFactory()
    gf.generate_images()
    gf.generate_gif()
