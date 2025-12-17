import glob
import os
import re
import random

from PIL import Image
import numpy as np
import mrcfile

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.v2 as v2
from torchvision.utils import save_image

from starparser import fileparser

from data.ice_train import (
    virtual_metadata,
    virtual_test,
    virtual_exclude3402,
    ppplite,
    ppp_metadata,
    virtual_draw,
)


# c, h, w numpy
def aug_np3(img, flip_h, flip_w, transpose):
    if flip_h:
        img = img[:, ::-1, :]
    if flip_w:
        img = img[:, :, ::-1]
    if transpose:
        img = np.transpose(img, (0, 2, 1))

    return img


def crop_np3(img, patch_size, position_h, position_w):
    return img[
        :, position_h: position_h + patch_size, position_w: position_w + patch_size
    ]


# Dataloader for denoise/seg model inference
# return img after z-score processing.


# Dataloader for denoise/seg model inference,
# work in three modes
# 1 denoise or segment any datasets without metric (default)
# 2 segment cryoPPP datasets  and metric with cryoPPP
# (root_dir = cryoppp_dir ,cryoppp_metric=True)
# 3 metric other tools' results with cryoPPP
# (root_dir = cryoppp_dir , cryoppp_metric= True, star_metric= True)


class parseek_inference(Dataset):
    def __init__(
        self,
        id,
        root_dir="",
        sub_dir="",
        cryoppp_dir="",
        img_format="",
        target_size=1024,
        metric_dir="ground_truth/particle_coordinates",  # cryoppp ground truth directory
        star_dir="",
        cryoppp_metric=False,
        star_metric=False,
        N=-1
    ):
        super(parseek_inference, self).__init__()
        self.id = id
        self.img_format = img_format

        if target_size > 0:
            self.target_size = target_size

        self.target_size = target_size

        self.norm_mode = getattr(__import__("src.aux"), "zscore")

        self.cryoppp_metric = cryoppp_metric
        self.star_metric = star_metric

        img_file_pattern = os.path.join(
            root_dir, id, sub_dir, "*." + self.img_format)
        img_paths_ = sorted(glob.glob(img_file_pattern))

        self.img_paths = []
        # skip Power spectra mrc(_PS.mrc) from relion motion correction result
        # and so on for cryoSPARC
        if self.img_format == "mrc":
            for x in img_paths_:
                if "_PS.mrc" in x:
                    # or "_patch_aligned.mrc" in x:
                    pass
                else:
                    self.img_paths.append(x)
        else:
            self.img_paths = img_paths_

        if N > 0:
            self.img_paths = img_paths_[0:min(N, len(self.img_paths))]
        else:
            self.img_paths = img_paths_

        if len(self.img_paths) == 0:
            raise FileNotFoundError("No files exist: " + img_file_pattern)

    def __getitem__(self, index):
        img_l = self._open_image(self.img_paths[index])
        img_L, meta = self.pad_and_normalize(img_l, self.target_size)

        # for user-defined dataset without metric
        if (not self.cryoppp_metric) and not (self.star_metric):
            file_name = os.path.basename(
                self.img_paths[index].split("." + self.img_format)[-2]
            )
            ppp_gt_csv = []
            return (img_L, file_name, meta, ppp_gt_csv)

        # for cryoppp dataset with cryoppp metric
        if self.cryoppp_metric and not (self.star_metric):
            file_name = os.path.basename(
                self.img_paths[index].split("." + self.img_format)[-2]
            )
            if self.id == "11056":
                file_name = re.sub("\d{21}_", "", file_name)
                file_name = re.sub("_patch_aligned", "", file_name)
            ppp_gt_csv = os.path.join(
                self.cryoppp_dir, self.id, self.metric_dir, file_name + ".csv"
            )
            return (img_L, file_name, meta, ppp_gt_csv)

        # compare other too result in star with cryoppp
        if self.cryoppp_metric and self.star_metric:
            file_name = os.path.basename(
                self.img_paths[index].split("." + self.img_format)[-2]
            )
            if self.id == "11056":
                file_name = re.sub("\d{21}_", "", file_name)
                file_name = re.sub("_patch_aligned", "", file_name)
            ppp_gt_csv = os.path.join(
                self.root_dir, self.id, self.metric_dir, file_name + ".csv"
            )
            star_result = os.path.join(
                self.root_dir, self.id, self.star_dir, file_name + ".star"
            )
            return (img_L, file_name, meta, ppp_gt_csv, star_result)

    def _open_image(self, path):
        match self.img_format:
            case "jpg" | "png" | "jpeg":
                # jpeg/jpg/png stored as uint8
                img = Image.open(path)
                img = np.asarray(img)
                img = img.astype(np.float32)
                img = img
            case "mrc":
                # sometimes mrc is stored in float16(mode=12)
                # convert to float32
                img = mrcfile.open(path, permissive=True)
                if int(img.header.mode) == 12:
                    img = img.data
                    img = img.astype(np.float32)
                else:
                    img = img.data
                    img = 255.0 * (img - img.min()) / (img.max() - img.min())
            case _:
                raise NotImplementedError(
                    "Format of micrographs is not supported!")

        # for h,w,c=3 png(uint8)
        # convert to (h, w)
        if len(img.shape) == 3:
            h, w, c = img.shape
            if c == 3:
                img = img[:, :, 0]
        img = img[np.newaxis, :]

        # (1, h ,w)
        return torch.Tensor(img)

    def pad_and_normalize(self, img, target_size=1024):
        _, h, w = img.shape

        if target_size > 0:
            scale = min(h, w) / target_size

            new_h = int(h / scale)
            new_w = int(w / scale)
            pad_h = target_size - new_h
            pad_w = target_size - new_w

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            # Resize the min(h,w) to target_size (h', t_size) or (t_size, w')
            tr = torch.nn.Sequential(
                v2.ToDtype(torch.float32),
                v2.Resize(size=target_size),
            )

            meta = {
                "oh": h,
                "ow": w,
                "mean": img.mean(),
                "std": img.std(),
                "pad_top": pad_top,
                "pad_bottom": pad_bottom,
                "pad_left": pad_left,
                "pad_right": pad_right,
                "scale": scale,
                "target_size": target_size,
            }

            img_t = tr(img)
            img_t = (img_t - img_t.mean()) / (img.std())

        else:
            meta = {
                "oh": h,
                "ow": w,
                "mean": img.mean(),
                "std": img.std(),
                "pad_top": 0,
                "pad_bottom": 0,
                "pad_left": 0,
                "pad_right": 0,
                "scale": 1,
                "target_size": target_size,
            }
            tr = torch.nn.Sequential(
                v2.ToDtype(torch.float32)
            )
            img_t = tr(img)
            img_t = (img_t - img_t.mean()) / (img.std())

        return img_t, meta

    def __len__(self):
        return len(self.img_paths)


# Dataloader for training parseek denoise model
class train_denoise(Dataset):
    def __init__(
        self,
        id,
        patch_size=512,
        root_dir="",
        sub_dir="",
        img_format="",
    ):
        super(train_denoise, self).__init__()
        self.id = id
        self.img_format = img_format
        self.patch_size = patch_size

        img_file_pattern = os.path.join(
            root_dir, id, sub_dir, "*." + self.img_format)
        img_paths_ = sorted(glob.glob(img_file_pattern))

        self.img_paths = []
        # skip Power spectra mrc(_PS.mrc) from relion motion correction result
        if self.img_format == "mrc":
            for x in img_paths_:
                if "_PS.mrc" in x or "_patch_aligned.mrc" in x:
                    pass
                else:
                    self.img_paths.append(x)
        else:
            self.img_paths = img_paths_

        # FIXME restrict num of micrographs to 10 for testing
        self.img_paths = self.img_paths

        if len(self.img_paths) == 0:
            raise FileNotFoundError("No files exist: " + img_file_pattern)
        if len(self.img_paths) > 500:
            random.shuffle(self.img_paths)
            self.img_paths = self.img_paths[0:500]

    def __getitem__(self, index):
        index = index % len(self.img_paths)
        img_L, meta = self._open_image(self.img_paths[index])
        return img_L, meta

    def crop(self, img_L, img_H=None):
        C, H, W = img_L.shape
        position_H = random.randint(0, H - self.patch_size)
        position_W = random.randint(0, W - self.patch_size)

        patch_L = crop_np3(img_L, self.patch_size, position_H, position_W)
        if img_H is not None:
            patch_H = crop_np3(img_H, self.patch_size, position_H, position_W)
            return patch_L, patch_H
        else:
            return patch_L

    def crop_augment_N(self, img_L, img_H=None, N=1):
        C, H, W = img_L.shape
        # patch_L = []
        for i in range(0, N):
            # crop
            position_H = random.randint(0, H - self.patch_size)
            position_W = random.randint(0, W - self.patch_size)
            img = crop_np3(img_L, self.patch_size, position_H, position_W)
            # aug
            flip_h = random.random() > 0.5
            flip_w = random.random() > 0.5
            transpose = random.random() > 0.5
            img = aug_np3(img, flip_h, flip_w, transpose)
            img = np.float32(img)
            # patch_L.append(img)
            patch_L = img

        if img_H is not None:
            patch_H = crop_np3(img_H, self.patch_size, position_H, position_W)
            return patch_L, patch_H
        else:
            return patch_L

    def augment(self, img_L, img_H=None):
        flip_h = random.random() > 0.5
        flip_w = random.random() > 0.5
        transpose = random.random() > 0.5
        img_L = aug_np3(img_L, flip_h, flip_w, transpose)
        if img_H is not None:
            img_H = aug_np3(img_H, flip_h, flip_w, transpose)
            return img_L, img_H
        else:
            return img_L

    def _open_image(self, path):
        match self.img_format:
            case "jpg" | "png" | "jpeg":
                # jpeg/jpg/png stored as uint8
                img = Image.open(path)
                img = np.asarray(img)
                img = img.astype(np.float32)
                img = img
            case "mrc":
                # sometimes mrc is stored in float16(mode=12)
                # convert to float32
                img = mrcfile.open(path, permissive=True)
                if int(img.header.mode) == 12:
                    img = img.data
                    img = img.astype(np.float32)
                    img = 255.0 * (img - img.min()) / (img.max() - img.min())
                else:
                    img = img.data
                    img = 255.0 * (img - img.min()) / (img.max() - img.min())
            case _:
                raise NotImplementedError(
                    "Format of micrographs is not supported!")

        # for h,w,c=3 png(uint8)
        if self.img_format == "png" and len(img.shape) == 3:
            h, w, c = img.shape
            if c == 3:
                img = img[:, :, 0]
        # h,w -> 1,h,w
        img = img[np.newaxis, :]

        if self.id == "10017":
            img = img[:, 256: 4096 - 256, 256: 4096 - 256]

        # crop and augment first ; scale ; zscore
        img_L = self.crop(img)
        img_L = self.augment(img_L)
        img_L = np.float32(img_L)
        img_L = np.copy(img_L)
        img_L = img_L[np.newaxis, :]

        img_L = torch.Tensor(img_L)
        # tr = v2.Resize((self.patch_size//4, self.patch_size//4))
        # img_L = tr(img_L)
        global_mean = img_L.mean()
        global_std = img_L.std()
        img_L = (img_L - global_mean) / global_std
        meta = {
            "global_mean": global_mean,
            "global_std": global_std,
        }

        # zscore ; crop and augment
        # img_L = (img - img.mean()) /img.std()
        # img_L = self.crop(img_L)
        # img_L = self.augment(img_L)
        # img_L = np.float32(img_L)
        # img_L = np.copy(img_L)

        # # to (n,c,h,w)
        # img_L = img_L[np.newaxis, :]
        # meta = {
        #     "global_mean": img.mean(),
        #     "global_std": img.std(),
        # }
        # img_L = torch.Tensor(img_L)

        return img_L, meta

    def __len__(self):
        return len(self.img_paths)


virtual_train = virtual_metadata
virtual_train = virtual_test
ppp = ppplite


# Dataloader for training parseek seg model
class train_seg(Dataset):
    def __init__(
        self,
        simu_micrograph_dir="",
        real_micrograph_dir="",
        img_format="mrc",
        target_size=1024,
        ratio=0.5,
        mask_mode="circle",
    ):
        """
        patch_size :>1 or -1 for full image
        img_format : mrc | jpg | png
        mask_mode : circle | dilate

        real_micrograph for cryoppp
        """
        super(train_seg, self).__init__()
        self.simu_micrograph_dir = simu_micrograph_dir
        self.real_micrograph_dir = real_micrograph_dir
        self.target_size = target_size
        self.ratio = ratio
        self.img_format = img_format
        self.mask_mode = mask_mode

        self.dynamic_preprocess = False

        self.simu_stars = []
        self.real_csvs = []
        self.micrographs = []

        self.rotate_star = {}
        self.rotate_projections = {}

        self.simu_emdb_id = virtual_train.keys()
        self.real_emdb_id = ppp.keys()

        self.simu_micrographs = []
        self.real_micrographs = []

        simu_micrographs_ = []
        real_micrographs_ = []

        if self.simu_micrograph_dir == "":
            raise FileNotFoundError("simu_micrograph_dir can not be empty!")

        for emdb_id in self.simu_emdb_id:
            simu_thisId = sorted(
                glob.glob(
                    os.path.join(
                        self.simu_micrograph_dir,
                        emdb_id,
                        "structure_set_1/*." + self.img_format,
                    ),
                    recursive=True,
                )
            )
            simu_micrographs_ += simu_thisId

        for emdb_id in self.simu_emdb_id:
            simu_thisId = sorted(
                glob.glob(
                    os.path.join(
                        self.simu_micrograph_dir, emdb_id, "structure_set_1/*.star"
                    ),
                    recursive=True,
                )
            )
            self.simu_stars += simu_thisId

        for mic in simu_micrographs_:
            mic_name = os.path.basename(mic).split("." + self.img_format)[0]
            mic_id = mic.split("/")[-3]
            star = os.path.join(
                self.simu_micrograph_dir, mic_id, "structure_set_1", mic_name + ".star"
            )

            if star in self.simu_stars and os.path.exists(star):
                # particles_gt, metadata = fileparser.getparticles(star)
                # TODO filter mic with 20-500 particles
                # if particles_gt._rlnCoordinateX.shape[0] >=20 and particles_gt._rlnCoordinateX.shape[0]<= 500 :
                #   self.micrographs.append(mic)
                self.simu_micrographs.append(mic)
            else:
                print(mic, star, "Star file is not existed.")

        if self.real_micrograph_dir != "":
            for emdb_id in self.real_emdb_id:
                real_thisId = sorted(
                    glob.glob(
                        os.path.join(
                            self.real_micrograph_dir,
                            ppp[emdb_id],
                            "micrographs/*." + self.img_format,
                        ),
                        recursive=True,
                    )
                )
                real_micrographs_ += real_thisId

                for emdb_id in self.real_emdb_id:
                    real_thisId = sorted(
                        glob.glob(
                            os.path.join(
                                self.real_micrograph_dir,
                                ppp[emdb_id],
                                "ground_truth/particle_coordinates/*.csv",
                            ),
                            recursive=True,
                        )
                    )
                    self.real_csvs += real_thisId

            for mic in real_micrographs_:
                mic_name = os.path.basename(mic).split(
                    "." + self.img_format)[0]
                mic_id = mic.split("/")[-3]
                csv = os.path.join(
                    self.real_micrograph_dir,
                    mic_id,
                    "ground_truth/particle_coordinates",
                    mic_name + ".csv",
                )
                if csv in self.real_csvs and os.path.exists(csv):
                    # particles_gt = pd.read_csv(csv)

                    # TODO filter mic with 20-500 particles
                    # if particles_gt._rlnCoordinateX.shape[0] >=20 and particles_gt._rlnCoordinateX.shape[0]<= 500 :
                    #   self.micrographs.append(mic)
                    self.real_micrographs.append(mic)
                else:
                    print(mic, csv, "tune csv file is not existed.")

    def get_groundtruth_mask(self, img, particles_gt, diameter):
        gt = np.full(img.shape, False)
        _, h, w = img.shape
        gt = np.full((h, w), False)

        # virtual ice generate default box =300
        rx, ry = np.ogrid[0:diameter, 0:diameter]
        cx = diameter // 2
        cy = diameter // 2

        dist_from_center = (rx - cx) ** 2 + (ry - cy) ** 2
        mask = np.where(
            dist_from_center <= np.power(
                diameter * self.ratio, 2) / 2, True, False
        )

        particles_gt._rlnCoordinateX = (
            particles_gt._rlnCoordinateX).astype("int")
        particles_gt._rlnCoordinateY = (
            particles_gt._rlnCoordinateY).astype("int")

        for index, row in particles_gt.iterrows():
            # print(rows._rlnCoordinateX,rows._rlnCoordinateY)
            # swap x and y for star generated by virtual Ice
            real_center_x = row._rlnCoordinateY
            real_center_y = row._rlnCoordinateX
            if (
                real_center_x - cx >= 0
                and real_center_x + cx <= h
                and real_center_y - cy >= 0
                and real_center_y + cy <= w
            ):
                tg = np.ma.mask_or(
                    gt[
                        real_center_x - cx: real_center_x + cx,
                        real_center_y - cy: real_center_y + cy,
                    ],
                    mask,
                )
                gt[
                    real_center_x - cx: real_center_x + cx,
                    real_center_y - cy: real_center_y + cy,
                ] = tg

        return torch.tensor(gt, dtype=torch.float32).reshape((1, h, w))

    def __getitem__(self, index):
        mic = self.simu_micrographs[index]
        mic_name = os.path.basename(mic).split("." + self.img_format)[0]
        mic_id = mic.split("/")[-3]

        diameter = virtual_train[mic_id]

        star = os.path.join(
            self.simu_micrograph_dir, mic_id, "structure_set_1", mic_name + ".star"
        )

        assert os.path.exists(mic)
        assert os.path.exists(star)

        img, _ = self._open_image(mic)
        img = v2.RandomVerticalFlip(p=1)(img)

        # img = normalize(img)[0]

        if torch.isnan(img).any():
            save_image(
                img,
                "./logs/seg_ice_tune/0%05d" % (index) + mic_name + ".jpg",
            )

        _, h, w = img.shape
        HW = min(h, w)

        particles_gt, metadata = fileparser.getparticles(star)

        gt = self.get_groundtruth_mask(img, particles_gt, diameter)

        ci, cj, ch, cw = v2.RandomCrop.get_params(img, output_size=(HW, HW))

        img_crop = torchvision.transforms.functional.crop(img, ci, cj, ch, cw)
        gt_crop = torchvision.transforms.functional.crop(gt, ci, cj, ch, cw)

        tr = v2.Resize((self.target_size, self.target_size))
        img_crop = tr(img_crop)
        gt_crop = tr(gt_crop)

        # random horizontally/vertical  flip
        flip_flag = np.random.randint(0, 2, 2)
        if flip_flag[0] == 1:
            img_crop = torchvision.transforms.functional.hflip(img_crop)
            gt_crop = torchvision.transforms.functional.hflip(gt_crop)

        if flip_flag[1] == 1:
            img_crop = torchvision.transforms.functional.vflip(img_crop)
            gt_crop = torchvision.transforms.functional.vflip(gt_crop)

        # random rot
        rot_degree = np.random.randint(0, 4, 1)[0]

        assert img_crop.shape == (1, self.target_size, self.target_size)
        assert gt_crop.shape == (1, self.target_size, self.target_size)
        img_crop = torch.rot90(img_crop, k=rot_degree, dims=(1, 2))
        gt_crop = torch.rot90(gt_crop, k=rot_degree, dims=(1, 2))

        img_crop = (img_crop - img_crop.mean()) / img_crop.std()
        return (
            img_crop,
            gt_crop,
            diameter,
            mic_id,
            self.dynamic_preprocess,
        )

    def _open_image(self, path):
        match self.img_format:
            case "jpg" | "png" | "jpeg":
                # jpeg/jpg/png stored as uint8
                img = Image.open(path)
                img = np.asarray(img)
                img = img.astype(np.float32)
                img = img
            case "mrc":
                # sometimes mrc is stored in float16(mode=12)
                # convert to float32
                img = mrcfile.open(path, permissive=True)
                if int(img.header.mode) == 12:
                    img = img.data
                    img = img.astype(np.float32)
                    img = 255.0 * (img - img.min()) / (img.max() - img.min())
                else:
                    img = img.data
                    img = 255.0 * (img - img.min()) / (img.max() - img.min())
            case _:
                raise NotImplementedError(
                    "Format of micrographs is not supported!")

        # for h,w,c=3 png(uint8)
        if self.img_format == "png" and len(img.shape) == 3:
            h, w, c = img.shape
            if c == 3:
                img = img[:, :, 0]
        # h,w -> 1,h,w
        if len(img.shape) == 2:
            img = img[np.newaxis, :]

        # if self.id == "10017":
        #     img = img[:, 256 : 4096 - 256, 256 : 4096 - 256]

        global_mean = img.mean()
        global_std = img.std()

        img_L = img
        img_L = np.float32(img_L)
        img_L = np.copy(img_L)

        # to (c,h,w)
        meta = {
            "global_mean": global_mean,
            "global_std": global_std,
        }
        return torch.Tensor(img_L), meta

    def __len__(self):
        return len(self.simu_micrographs)
