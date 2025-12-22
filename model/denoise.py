# Modified from https://github.com/nagejacob/SpatiallyAdaptiveSSID.git
from model.base import BaseModel
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torchvision.utils import save_image
from src.aux import zscore
import kornia
from typing import Tuple


def std(img, window_size=21):
    assert window_size % 2 == 1
    pad = window_size // 2

    # calculate std on the mean image of the color channels
    img = torch.mean(img, dim=1, keepdim=True)
    N, C, H, W = img.shape
    img = nn.functional.pad(img, [pad] * 4, mode="reflect")
    img = nn.functional.unfold(img, kernel_size=window_size)
    img = img.view(N, C, window_size * window_size, H, W)
    img = img - torch.mean(img, dim=2, keepdim=True)
    img = img * img
    img = torch.mean(img, dim=2, keepdim=True)
    img = torch.sqrt(img)
    img = img.squeeze(2)
    return img


class Clahe(torch.nn.Module):
    def __init__(
        self, clip_limit: int | float = 16, grid_size: Tuple[int, int] = (16, 16)
    ) -> None:
        super().__init__()
        self.clip_limit, self.grid_size = float(clip_limit), grid_size

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return kornia.enhance.equalize_clahe(img, self.clip_limit, self.grid_size)

    def __repr__(self) -> str:
        return "{}(clip_limit={}, tile_grid_size={})".format(
            self.__class__.__name__, self.clip_limit, self.grid_size
        )


class parseek_denoise(BaseModel):
    def __init__(self, opt):
        super(parseek_denoise, self).__init__(opt)
        self.stage = None
        self.criteron = nn.L1Loss(reduction="mean")
        self.optimizer_BNN = torch.optim.Adam(
            self.networks["BNN"].parameters(), lr=opt["lr"]
        )
        self.optimizer_UNet = torch.optim.Adam(
            self.networks["UNet"].parameters(), lr=opt["lr"]
        )

    def train_step(self, data, global_mean, global_std):
        self.update_stage()

        input = data

        match self.stage:
            case "BNN":
                self.networks["BNN"].train()
                BNN = self.networks["BNN"](input)
                self.loss = self.criteron(BNN, input)
                self.optimizer_BNN.zero_grad()
                self.loss.backward()
                self.optimizer_BNN.step()

            case "UNet":
                self.networks["BNN"].eval()
                self.networks["UNet"].train()
                with torch.no_grad():
                    BNN = self.networks["BNN"](input)
                UNet = self.networks["UNet"](input)
                self.loss = self.criteron(BNN, UNet)
                self.optimizer_UNet.zero_grad()
                self.loss.backward()
                self.optimizer_UNet.step()

        self.iter += 1

        return self.loss.item() * 1.0

    def validation_step(self, data, global_mean, global_std):
        self.update_stage()
        input = data
        input_z, input_m, input_s = zscore(input)

        if self.stage == "BNN":
            self.networks["BNN"].eval()
            with torch.no_grad():
                output = self.networks["BNN"](input_z)
                output = output * global_std + global_mean
        elif self.stage == "UNet":
            self.networks["UNet"].eval()
            with torch.no_grad():
                output = self.networks["UNet"](input_z)
                output = output * global_std + global_mean

        return output

    def validation_save(self, data, output):
        img = data.squeeze(0)  # /255.0
        img = (img - img.min()) / (img.max() - img.min())
        output = output  # / 255.0
        output = (output - output.min()) / (output.max() - output.min())
        save_path = os.path.join(
            self.opt["log_dir"],
            "net_%s_epoch_%08d_%08d.png" % (self.stage, self.epoch, self.iter),
        )
        save_image([img, output.squeeze(0)], save_path, normalize=False)

    def validation_step_manual(self, data, stage):
        input = data

        if stage == "BNN":
            self.networks["BNN"].eval()
            with torch.no_grad():
                output = self.networks["BNN"](input)
        elif stage == "UNet":
            self.networks["UNet"].eval()
            with torch.no_grad():
                output = self.networks["UNet"](input)

        return output

    def save_net(self):
        if self.stage == "BNN":
            net = self.networks["BNN"]
        elif self.stage == "UNet":
            net = self.networks["UNet"]

        if isinstance(net, DataParallel):
            net = net.module
        torch.save(
            net.state_dict(),
            os.path.join(
                self.opt["log_dir"], "net_%s_epoch_%05d.pth" % (self.stage, self.epoch)
            ),
        )

    def save_model(self):
        if self.stage == "BNN":
            save_dict = {
                "iter": self.iter,
                "optimizer_BNN": self.optimizer_BNN.state_dict(),
                "BNN": self.networks["BNN"].state_dict(),
            }
        elif self.stage == "UNet":
            save_dict = {
                "iter": self.iter,
                "optimizer_UNet": self.optimizer_UNet.state_dict(),
                "BNN": self.networks["BNN"].state_dict(),
                "UNet": self.networks["UNet"].state_dict(),
            }
        torch.save(
            save_dict,
            os.path.join(self.opt["log_dir"], "model_epoch_%05d.pth" % self.epoch),
        )

    def load_model(self, path):
        load_dict = torch.load(path)
        self.iter = load_dict["iter"]
        self.update_stage()
        if self.stage == "BNN":
            self.networks["BNN"].load_state_dict(load_dict["BNN"])
        elif self.stage == "UNet":
            self.networks["BNN"].load_state_dict(load_dict["BNN"])
            self.networks["UNet"].load_state_dict(load_dict["UNet"])
        else:
            raise NotImplementedError

    def update_stage(self):
        if self.epoch <= self.opt["BNN_epoch"]:
            self.stage = "BNN"
        else:
            self.stage = "UNet"
