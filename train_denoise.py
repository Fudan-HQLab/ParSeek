import os
import argparse

from torch.utils.data import DataLoader
from utils.build import build
from utils.io import log
from utils.option import parse, recursive_log


def collate_custom(data):
    crop_img = data[0][0]
    global_mean = data[0][1]
    global_std = data[0][2]
    return crop_img, global_mean, global_std


def collate_custom2(data):
    img = data[0][0]
    meta = data[0][1]
    return img, meta


def main(opt):
    train_dataset_opt = opt["train_datasets"]
    TrainDataset = getattr(__import__("dataset"), train_dataset_opt["dataset"])
    train_set = build(TrainDataset, train_dataset_opt["args"])
    train_loader = DataLoader(
        train_set,
        batch_size=train_dataset_opt["batch_size"],
        shuffle=True,
        num_workers=28,
        drop_last=True,
        collate_fn=collate_custom2,
        pin_memory=False,
    )

    Model = getattr(__import__("model"), opt["model"])
    model = Model(opt)

    model.data_parallel()

    epoch = opt["epoch"]
    start_epoch = opt["start_epoch"]
    model.iter = start_epoch * len(train_loader)

    if "resume_from" in opt:
        model.load_model(opt["resume_from"])

    def train_step(img, global_mean, global_std):
        model.train_step(img)
        # while trianing on cryoppp, one epoch has about 300 iters(batchsize=1)
        # and about 80 iters for EMPIAR-10017
        if model.epoch % opt["print_every"] == 0:
            model.log()

        if model.epoch % opt["save_every"] == 0:
            model.save_net()
            model.save_model()

        if model.epoch % opt["validate_every"] == 0:
            message = "iter: %d, " % model.iter
            validate_output = model.validation_step(img, global_mean, global_std)
            model.validation_save(img, validate_output)
            log(opt["log_file"], message + "\n")

    current_epoch = start_epoch
    while current_epoch - start_epoch <= epoch:
        epoch_loss = 0.0
        for img, meta in train_loader:
            img = img.cuda()
            epoch_loss += model.train_step(img, meta["global_mean"], meta["global_std"])
            if model.iter % opt["print_every"] == 0:
                model.log()

            # when train dataset is too large, validate once every epoch
            if len(train_loader) > 300:
                if model.iter % opt["validate_every"] == 0:
                    validate_output = model.validation_step(
                        img, meta["global_mean"], meta["global_std"]
                    )
                    model.validation_save(img, validate_output)
            else:
                if model.iter % len(train_loader) == 0:
                    validate_output = model.validation_step(
                        img, meta["global_mean"], meta["global_std"]
                    )
                    model.validation_save(img, validate_output)

        epoch_loss /= len(train_loader)

        if model.epoch % opt["save_every"] == 0:
            model.save_net()
            model.save_model()

        log(opt["log_file"], "epoch: %d, mean loss: %f\n" % (current_epoch, epoch_loss))
        current_epoch += 1
        model.epoch += 1
    if current_epoch == epoch + start_epoch:
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ParSeek denoiser")
    parser.add_argument(
        "--config", type=str, default="options/example_train_denoise.json"
    )
    argspar = parser.parse_args()

    opt = parse(argspar.config)
    if os.path.exists(os.path.dirname(opt["log_dir"])):
        print(opt["log_dir"])
        if not os.path.exists(opt["log_dir"]):
            os.mkdir(opt["log_dir"])

    recursive_log(opt["log_file"], opt)

    main(opt)
