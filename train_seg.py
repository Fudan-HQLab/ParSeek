import argparse
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.build import build
from utils.io import log
from utils.option import parse, recursive_log
from torchvision.utils import save_image
from torchvision.transforms import v2
import torch
import mrcfile
import os

def get_dataloader(dataset, num_samples, batch_size=8, shuffle=True):
    if shuffle:
        indices = torch.randperm(len(dataset))[:num_samples]
    else:
        indices = torch.arange(num_samples)
    sampler = SubsetRandomSampler(indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=28,
        sampler=sampler,
        drop_last=True,
        pin_memory=True,
    )


def main(opt):
    train_dataset_opt = opt["train_datasets"]
    TrainDataset = getattr(__import__("dataset"), train_dataset_opt["dataset"])
    train_set = build(TrainDataset, train_dataset_opt["args"])
    # Initialize num_samples_per_epoch to 0 (original value: 750); this variable controls the number of samples processed per training epoch.
    # When set to 0, it bypasses the fixed sample limit (750) and triggers iteration over the entire training dataset.
    num_samples_per_epoch = 0
    # iterate on full dataset
    if num_samples_per_epoch == 0:
        train_loader = DataLoader(
            train_set,
            batch_size=train_dataset_opt["batch_size"],
            shuffle=True,
            # num_workers=train_dataset_opt["batch_size"],
            num_workers=28,
            drop_last=True,
            pin_memory=True,
        )


    Model = getattr(__import__("model"), opt["model"])
    model = Model(opt)
    recursive_log(opt["log_file"], "model params:" + str(model.get_param_count()))

    emd_set = set()

    device = opt["device"]

    model.data_parallel([0])
    # epoch is 1-index based
    epoch = opt["epoch"]
    start_epoch = opt["start_epoch"]

    if num_samples_per_epoch == 0:
        model.iter = start_epoch * len(train_loader)
    else:
        model.iter = start_epoch * num_samples_per_epoch
    if "resume_from" in opt:
        model.load_model(opt["resume_from"])
    
    #Switch to your own Path
    '''validation_img = mrcfile.open(
        "/Your/Path/To/mrc/File"
    ).data.copy()
    validate_x = torch.tensor(validation_img)
    validate_x = validate_x.unsqueeze(0).unsqueeze(0)
    validate_x = v2.Resize((1024, 1024))(validate_x)
    validate_x = (validate_x - validate_x.mean()) / validate_x.std()
    validate_x = validate_x.cuda()'''

    def train_step(data, GT, particleSize, mic_id, dp):
        model.train_step(data, GT, particleSize, mic_id, dp)

        if model.iter % opt["print_every"] == 0:
            model.log()

        if model.iter % opt["save_every"] == 0:
            model.save_net()
            model.save_model()

        if model.iter == opt["num_iters"]:
            model.save_net()
            exit()

    def test_step(X, Y, mic):
        if mic not in emd_set:
            emd_set.add(mic)
            save_image(X, os.path.join(opt["log_dir"], mic + ".jpg"))
            save_image(Y, os.path.join(opt["log_dir"], mic + "_mask.jpg"))
        else:
            pass

    current_epoch = start_epoch
    while current_epoch - start_epoch <= epoch:
    # Execute model's validation step with preprocessed validation image
    # This step typically computes validation metrics (e.g., loss/accuracy) or generates inference results
        #model.validation_step(validate_x)
        # iterate on part of dataset
        if num_samples_per_epoch:
            train_loader = get_dataloader(
                train_set,
                num_samples_per_epoch,
                batch_size=train_dataset_opt["batch_size"],
                shuffle=True,
            )

        mic_loss = {}
        epoch_loss = 0.0

        for X, Y, particleSize, mic, dynamic_preprocess in train_loader:
            X = X.to(device)
            Y = Y.to(device)
            this_loss = model.train_step(X, Y, particleSize, mic, dynamic_preprocess)
            epoch_loss += this_loss.mean()
            if model.iter % opt["print_every"] == 0:
                model.log()
            for i in range(X.shape[0]):
                if mic[i] in mic_loss.keys():
                    mic_loss[mic[i]]["loss"] += this_loss[i]
                    mic_loss[mic[i]]["count"] += 1
            else:
                mic_loss[mic[i]] = {"loss": this_loss[i], "count": 1, "mean": 0.0}

        epoch_loss /= len(train_loader)
        model.save_net()
        model.save_model()
        for k in mic_loss.keys():
            mic_loss[k]["mean"] = mic_loss[k]["loss"] / mic_loss[k]["count"]
        log(opt["log_file"], "epoch: %d, loss: %f\n" % (current_epoch, epoch_loss))
        log(opt["log_file"], "epoch: %d, \n" % (current_epoch))
        log(opt["log_file"], "%s \n" % (mic_loss))
        current_epoch += 1
        model.epoch += 1
    if current_epoch == epoch + start_epoch:
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ParSeek SEG")
    parser.add_argument("--config", type=str, default="options/example_train_seg.json")
    argspar = parser.parse_args()

    opt = parse(argspar.config)
    if os.path.exists(os.path.dirname(opt["log_dir"])):
        print(opt["log_dir"])
        if not os.path.exists(opt["log_dir"]):
            os.mkdir(opt["log_dir"])

    recursive_log(opt["log_file"], opt)

    main(opt)
