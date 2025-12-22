import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torchvision.utils import save_image
import torchvision.transforms.v2 as v2

from torchinfo import summary

from model.base import BaseModel
from src.aux import normalize, zscore


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size, w // block_size)


def depth_to_space(x, block_size):
    return torch.nn.functional.pixel_shuffle(x, block_size)


def simpleLoss(pred: torch.tensor, target: torch.tensor):
    return target - pred


# https://github.com/milesial/Pytorch-UNet/blob/master/utils/dice_score.py
def dice_coeff(
    input: torch.tensor,
    target: torch.tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    # Average of Dice coefficient for all batches, or for a single mask
    # assert input.size() == target.size()
    # assert input.dim() == 3 or not reduce_batch_first
    # sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    sum_dim = (2, 3)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(
    input: torch.tensor,
    target: torch.tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    # Average of Dice coefficient for all classes
    return dice_coeff(
        input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon
    )


def dice_loss(pred: torch.tensor, target: torch.tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(pred, target, reduce_batch_first=False)


def dice_score(pred: torch.tensor, target: torch.tensor):
    smooth = 0.0001
    if target.dim() == 3:
        target = target.unsqueeze(1)
    # pred = pred.view(-1)
    # target = target.contiguous().view(-1)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    score = (2 * intersection + smooth) / (union + intersection + smooth)
    return score


# https://discuss.pytorch.org/t/what-loss-function-for-binary-unet/143529/2
def bce_dice(pred: torch.tensor, target: torch.tensor):
    pi = (target == 0.0).sum() / (target.sum() + 1e-6)
    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pi)(pred, target)
    dice_loss = 1 - dice_score(nn.Sigmoid()(pred), target)
    return (bce_loss + dice_loss) / 2.0


def focal(
    pred: torch.tensor, target: torch.tensor, alpha=0.75, gamma=2.0, reduction="mean"
):
    probs = nn.Sigmoid()(pred)
    pt = torch.where(target == 1, probs, 1 - probs)
    focal_weight = (1 - pt) ** gamma
    alpha_t = torch.where(target == 1, alpha, 1 - alpha)
    bce_loss = nn.BCEWithLogitsLoss(reduction="none")(pred,target)
    focal_loss = alpha_t * focal_weight * bce_loss

    match reduction:
        case "mean":
            return focal_loss.mean()
        case "sum":
            return focal_loss.sum()
        case _:
            return focal_loss


def local_score(input: torch.tensor, target: torch.tensor, particleSize, locations):
    """
    mask value range from radius**2(center) to 0(edge)
    """
    local_pick_loss = 0.0
    count = 0
    power_particleSize = particleSize**2
    for x, y in locations:
        a = input[x : x + particleSize, y : y + particleSize]
        b = target[x : x + particleSize, y : y + particleSize]
        count + 1
        local_pick_loss += (
            nn.functional.MSELoss(a, b, reduction="sum") / power_particleSize
        )

    local_pick_loss = local_pick_loss / count
    return local_pick_loss


class parseek_seg_model(BaseModel):
    def __init__(self, opt):
        super(parseek_seg_model, self).__init__(opt)
        self.opt = opt
        self.optimizer = torch.optim.AdamW(
            self.networks["SEG"].parameters(), lr=opt["lr"], weight_decay=0.01
        )

        summary(self.networks["SEG"], input_size=(1, 1, 1024, 1024))

    def train_step(self, input, GT, p, mic, dp):
        self.networks["SEG"].train()
        self.optimizer.zero_grad()

        SEG = self.networks["SEG"](input)

        loss = bce_dice(pred=SEG, target=GT)

        self.loss = loss.mean()

        self.loss.backward()

        self.optimizer.step()

        if self.iter % self.opt["val_every"] == 0:
            save_image(
                torch.cat(
                    [
                        normalize(input[0:1])[0],
                        nn.Sigmoid()(SEG[0:1]),
                        GT[0:1] * normalize(input[0:1])[0],
                    ]
                ),
                self.opt["log_dir"] + "/%08d_%s" % (self.iter, str(mic[0])) + ".jpg",
            )
            save_image(
                normalize(input[0:1])[0],
                self.opt["log_dir"] + "/%08d_%s_o" % (self.iter, str(mic[0])) + ".jpg",
            )

            save_image(
                normalize(GT[0:1])[0],
                self.opt["log_dir"] + "/%08d_%s_m" % (self.iter, str(mic[0])) + ".jpg",
            )

        if torch.isnan(self.loss):
            save_image(
                torch.cat(
                    [normalize(input)[0], nn.Sigmoid()(SEG), GT * normalize(input)[0]]
                ),
                self.opt["log_dir"] + "/nan_%08d_%s" % (self.iter, mic) + ".jpg",
            )
            exit()

        self.iter += 1
        return loss.detach().cpu().numpy()

    def validation_step(self, data):
        with torch.no_grad():
            seg_result = self.networks["SEG"](data)

        seg_result = torch.sigmoid(seg_result)
        seg_result = (seg_result - seg_result.min()) / (
            seg_result.max() - seg_result.min()
        )
        save_image(
            [normalize(data)[0].squeeze(0), seg_result.squeeze(0)],
            os.path.join(self.opt["log_dir"], "vali_epoch_%05d.jpg" % self.epoch),
        )
        return 0

    def save_net(self):
        net = self.networks["SEG"]

        if isinstance(net, DataParallel):
            net = net.module

        torch.save(
            net.state_dict(),
            os.path.join(
                self.opt["log_dir"], "net_%s_epoch_%05d.pth" % ("SEG", self.epoch)
            ),
        )

    def save_model(self):
        save_dict = {
            "iter": self.iter,
            "optimizer": self.optimizer.state_dict(),
            "SEG": self.networks["SEG"].state_dict(),
        }

        torch.save(
            save_dict,
            os.path.join(self.opt["log_dir"], "model_epoch_%05d.pth" % self.epoch),
        )

    def load_model(self, path):
        load_dict = torch.load(path)
        self.iter = load_dict["iter"]
        self.optimizer.load_state_dict(load_dict["optimizer"])
        self.networks["SEG"].load_state_dict(load_dict["SEG"])

    def get_param_count(self):
        pp = 0
        for p in list(self.networks["SEG"].parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp
