import os
import argparse

import numpy as np
import pandas as pd
import scipy
import queue

import torch
import torchvision.transforms.v2 as v2
from torchvision.utils import save_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure
from torchinfo import summary

from PIL import Image
import mrcfile
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

# from torchmetrics.functional.image import psnr
from kornia.enhance import equalize_clahe
from kornia.metrics import psnr
import kornia.geometry.transform as kgt
import statistics as st

# load sam model to predict unet segment results
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import tqdm
from starparser import fileparser
import matplotlib.pyplot as plt
import trackpy as tp

from utils.aux import normalize, zscore
from utils.build import build
from utils.io import log
from utils.option import parse, recursive_log

sam_model = sam_model_registry["vit_h"](
    checkpoint="/data/parsed2/git/pretrained_models/sam_vit_h_4b8939.pth"
)
sam_model.to(device="cuda:0")


def draw_bounding_boxes_wrapper(img, boxes, colors, width=3):
    if img.min() >= 0.0 and img.max() <= 1.0:
        img = img
    else:
        img = normalize(img)[0]

    img = img * 255
    img = torch.round(img)
    img = img.to(torch.uint8)
    img_N = torch.cat((img, img, img), dim=1)
    if boxes == []:
        return img_N.squeeze(0) / 255.0
    else:
        res = draw_bounding_boxes(
            image=img_N.squeeze(0), boxes=boxes, colors=colors, width=width
        )
        return res / 255.0


def draw_segmentation_masks_wrapper(img, mask, alpha=0.6, colors=""):
    if img.min() >= 0.0 and img.max() <= 1.0:
        img = img
    else:
        img = normalize(img)[0]

    img = img * 255
    img = torch.round(img)
    img = img.to(torch.uint8)

    bi_mask = torch.where(mask > 0.5, True, False)
    img_N = torch.cat((img, img, img), dim=1)
    res = draw_segmentation_masks(
        img_N.squeeze(0), bi_mask.squeeze(0), alpha, colors=colors
    )
    return res / 255.0


def pick_metrics(
    img,
    img_name,
    pick_result,
    ppp_gtruth,
    diameter,
    ratio,
    invertx=False,
    inverty=False,
    swapxy=False,
    flag=0,
):
    t1 = pick_result.loc[:, ["x", "y"]].astype(np.float32)
    gt = ppp_gtruth.loc[:, ["X-Coordinate", "Y-Coordinate"]].astype(np.float32)

    n, c, h, w = img.shape

    if flag == 1:
        if invertx:
            t1["x"] = w - t1["x"]
        if inverty:
            t1["y"] = h - t1["y"]
    elif flag == 2:
        if invertx:
            gt["X-Coordinate"] = w - gt["X-Coordinate"]
        if inverty:
            gt["Y-Coordinate"] = h - gt["Y-Coordinate"]

    t1_point = np.array(t1[["x", "y"]])

    if swapxy:
        gt_point = np.array(gt[["Y-Coordinate", "X-Coordinate"]])
    else:
        gt_point = np.array(gt[["X-Coordinate", "Y-Coordinate"]])

    matr = scipy.spatial.distance.cdist(t1_point, gt_point, "euclidean")

    aiparse_pick_metric = np.where(matr < diameter * ratio)

    # TruePositive
    # to avoid recall > 1.0 (two picking pts match one gt pt)
    # counting gt pt to calculate precision
    TP_ = aiparse_pick_metric[1].tolist()
    # eliminate replicate gt pts
    TP_ = set(TP_)
    TP_ = list(TP_)
    TP = []
    for i in TP_:
        TP.append(matr[:, i].argmin())
    TP = set(TP)
    TP = list(TP)
    # FalsePositive
    FP = []
    for p in range(0, t1_point.shape[0]):
        if p not in TP:
            FP.append(p)

    FP = set(FP)
    FP = list(FP)

    particle_flag = []
    for i in range(0, t1_point.shape[0]):
        if i in TP:
            particle_flag.append(1)
        else:
            particle_flag.append(0)

    pick_result["IsParticle"] = particle_flag

    # FalseNegative
    FN = []
    for p in range(0, gt_point.shape[0]):
        if p not in aiparse_pick_metric[1].tolist():
            FN.append(p)

    FN = set(FN)
    FN = list(FN)

    TP_color = "green"
    FP_color = "red"
    FN_color = "blue"

    TP_colors = [TP_color] * len(TP)
    FP_colors = [FP_color] * len(FP)
    FN_colors = [FN_color] * len(FN)

    TP_particles_boxes = []
    FP_particles_boxes = []
    FN_particles_boxes = []

    particles_boxes = []
    colors = []

    for p in TP:
        TP_particles_boxes.append(
            boxes(t1_point[p][0], t1_point[p][1], diameter))

    for p in FP:
        FP_particles_boxes.append(
            boxes(t1_point[p][0], t1_point[p][1], diameter))

    for p in FN:
        FN_particles_boxes.append(
            boxes(gt_point[p][0], gt_point[p][1], diameter))

    particles_boxes = TP_particles_boxes + FP_particles_boxes + FN_particles_boxes
    particles_boxes = torch.tensor(particles_boxes)

    colors = TP_colors + FP_colors + FN_colors

    if t1.shape[0] > 0:
        metric_dict = {
            "mic": img_name,
            "TotalPicked": int(t1.shape[0]),
            "GT": int(gt.shape[0]),
            "TP": len(TP),
            "FP": len(FP),
            "FN": len(FN),
            "Precision": len(TP) / t1.shape[0],
            "Recall": len(TP) / gt.shape[0],
            "F1-score": 0.0,
        }
    else:
        metric_dict = {
            "mic": img_name,
            "TotalPicked": int(t1.shape[0]),
            "GT": int(gt.shape[0]),
            "TP": len(TP),
            "FP": len(FP),
            "FN": len(FN),
            "Precision": 0.0,
            "Recall": len(TP) / gt.shape[0],
            "F1-score": 0.0,
        }
    if not metric_dict["TP"] == 0:
        metric_dict["F1-score"] = (
            2.0
            * metric_dict["Precision"]
            * metric_dict["Recall"]
            / (metric_dict["Precision"] + metric_dict["Recall"])
        )

    log(
        opt["log_file"],
        ",%s,TotalPicked:%d,GroundTruth:%d,TruePicked:%d,Precision:%.2f,Recall:%.2f,F1-score:%.2f\n"
        % (
            img_name,
            metric_dict["TotalPicked"],
            metric_dict["GT"],
            metric_dict["TP"],
            metric_dict["Precision"],
            metric_dict["Recall"],
            metric_dict["F1-score"],
        ),
    )

    return pick_result, particles_boxes, colors, metric_dict


def check_on_edge(x, y, h, w, d):
    return (x > w - d) or (x < d) or (y > h - d) or (y < d)


def check_on_edge_pandas(x, h, w, r):
    "x.x for h, x.y for w"
    return (x.x > h - r) or (x.x < r) or (x.y > w - r) or (x.y < r)


def local_img(img, box):
    _, _, h, w = img.shape
    t = img[:, :, box[0]: box[2], box[1]: box[3]]
    # print(t.shape)
    return t


def local_rawmass(img, box):
    t = img[:, :, box[0]: box[2], box[1]: box[3]]
    return t.sum()


def boxes(cx, cy, d):
    "return x_left,y_left,x_right,y_right"
    return [cx - d // 2, cy - d // 2, cx + d // 2, cy + d // 2]


def pearson_coeff(x, y):
    return torch.corrcoef(torch.cat((x, y), dim=0))


def SSIM(x, y):
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    z = structural_similarity_index_measure(
        x, y, kernel_size=5, data_range=255.0)
    return z


def parse_pad_resize(img, h, w, target_size=1024):
    if h == w:
        tr = v2.Resize((1024, 1024))
        pad_h = 0
        pad_w = 0
        meta = {"scale": 1.0, "pad_w": pad_w,
                "pad_h": pad_h, "original_shape": (h, w)}
        return tr((img.squeeze(0))).unsqueeze(0), meta
    else:
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        tr = v2.Resize((new_h, new_w))

        resized = tr((img.squeeze(0))).unsqueeze(0)

        padded = torch.zeros(
            (1, target_size, target_size), dtype=torch.float32)
        pad_w = (target_size - new_w) // 2
        pad_h = (target_size - new_h) // 2
        padded[:, pad_h: pad_h + new_h, pad_w: pad_w + new_w] = resized
        meta = {
            "scale": scale,
            "pad_w": pad_w,
            "pad_h": pad_h,
            "original_shape": (h, w),
        }
        return padded, meta


def parse_unpad_resize(img, meta):
    scale = meta["scale"]
    pad_h = meta["pad_h"]
    pad_w = meta["pad_w"]
    h, w = meta["original_shape"]
    new_h = int(h * scale)
    new_w = int(w * scale)
    cropped = img[:, pad_h: pad_h + new_h, pad_w: pad_w + new_w]
    tr = v2.Resize(h, w)
    return tr((cropped)).unsqueeze(0)


def circular_mask(d):
    rx, ry = np.ogrid[0:d, 0:d]
    cx = d // 2
    cy = d // 2
    dist_from_center = (rx - cx) ** 2 + (ry - cy) ** 2
    mask = np.where(dist_from_center <= (d**2 / 4), 1, 0)
    return mask


def plot_mass(
    aiparse_output_root,
    name,
    raw_img,
    denoise_img,
    seg_img,
    diameter,
    particles,
    metric_particles,
    metric_colors,
):
    csv_file = os.path.join(aiparse_output_root, name + ".csv.gz")
    particles.to_csv(csv_file, compression="gzip")
    seg_mass = []
    tp_raw_mass = []
    for index, row in particles.iterrows():
        thisbox = boxes(int(row.x), int(row.y), diameter)
        rmass = local_rawmass(seg_img, thisbox)
        tp_raw_mass.append(row.raw_mass)
        seg_mass.append(rmass)

    n_bins = 64
    plt.rcParams["figure.figsize"] = (12.8, 7.2)
    fig, axs = plt.subplots(1, 4, tight_layout=False)
    axs[0].hist(np.array(seg_mass), bins=n_bins, density=True)
    axs[0].title.set_text("seg_mass")
    axs[1].hist(np.array(tp_raw_mass), bins=n_bins, density=True)
    axs[1].title.set_text("tp_raw_mass")

    # avearge brightness
    brightness = []
    coeff = []
    mask = circular_mask(diameter)
    colors = []
    for index, row in particles.iterrows():
        thisbox = boxes(int(row.x), int(row.y), diameter)
        this_brightness = (local_img(seg_img, thisbox) * mask).sum()
        brightness.append(this_brightness)
        coeff.append(
            SSIM(255 - local_img(seg_img, thisbox),
                 local_img(denoise_img, thisbox))
        )
        if row.IsParticle == 1:
            colors.append("green")
        else:
            colors.append("red")

    axs[2].hist(np.array(brightness), bins=n_bins, density=True)
    axs[2].title.set_text("brightness")
    axs[3].hist(np.array(coeff), bins=n_bins, density=True)
    axs[3].title.set_text("coeff")

    histfig_file = os.path.join(aiparse_output_root, name + ".png")
    plt.savefig(histfig_file)

    fig_scatter, axs = plt.subplots(1, 1)
    # while "blue" in metric_colors:
    #    metric_colors.remove("blue")
    axs.scatter(np.array(brightness), np.array(coeff), s=2, c=colors)
    scatter_file = os.path.join(aiparse_output_root, name + "_scatter.png")
    plt.savefig(scatter_file)


mask_generator = SamAutomaticMaskGenerator(
    sam_model, points_per_side=64, points_per_batch=512
)


def generate_sam_mask(
    img: torch.tensor,
    origin_img: torch.tensor,
    diameter: int,
    separation: float,
    pixelSize: float,
    threshold: float,
    cut_edge=False,
    flip_Y=True,
):
    # img (1,1,1024,1024) with sigmoid(wo normalize)
    # sam mask generator read in image (h, w, c)
    _, _, scale_h, scale_w = img.shape

    # NOTE filtering artifects
    img_filter_mask = torch.where(img > threshold, 1, 0)

    img = img * img_filter_mask

    predicted_mask = img.squeeze(0)
    predicted_mask = torch.cat((img, img, img), dim=1)
    predicted_mask = predicted_mask.squeeze(0).permute(1, 2, 0)

    sam_input = predicted_mask.cpu().numpy()

    masks = mask_generator.generate(sam_input)
    return masks


# TODO add a new picking fuction from
# Use trackpy to extract particles ,similiar to PARSED
# https://www.kaggle.com/code/hocop1/unet-character-detector/notebook
# trackpy -> (center x , center y)
def extract_dilate(
    img: torch.tensor,
    origin_img: torch.tensor,
    diameter: int,
    separation: float,
    pixelSize: float,
    threshold: float,
    cut_edge=False,
    flip_Y=True,
):
    # diameter / pixelSize -> pixel
    # for mask_ratio == 0.25, d = diameter //4= dima
    # tp.locate read in image with 0-255,uint8

    _, _, scale_h, scale_w = img.shape
    n, c, h, w = origin_img.shape
    # convert to uint8
    img = img * 255
    img = torch.round(img)
    img = img.to(torch.uint8)

    d = int((diameter))
    if h == w:
        scale_d = d * scale_h / h
    else:
        scale_d = d * scale_h / h + w * scale_w / w
        scale_d = d / 2

    scale_d = scale_d // 2 * 2 + 1

    img = img.to("cpu")
    img_np = img.numpy()

    particles = tp.locate(
        img_np,
        diameter=scale_d,
        separation=int(scale_d * separation // 2),
        minmass=0,
        percentile=0.5,
    )
    print(particles.shape)
    if flip_Y:
        particles.y = scale_h - particles.y

    particles.x = (particles.x).astype(np.int32)
    particles.y = (particles.y).astype(np.int32)

    particles.x = particles.x * w / scale_w
    particles.y = particles.y * h / scale_h

    particle_boxes = []
    color = "red"
    colors = []
    if cut_edge:
        particles_new = particles[
            ~particles.apply(
                check_on_edge_pandas, axis=1, h=w, w=h, r=diameter // 2 + diameter
            )
        ]
    else:
        particles_new = particles[
            ~particles.apply(check_on_edge_pandas, axis=1,
                             h=w, w=h, r=diameter // 2)
        ]

    for index, row in particles_new.iterrows():
        colors.append(color)
        thisbox = boxes(int(row.x), int(row.y), diameter)
        particle_boxes.append(thisbox)
        colors.append(color)

    particle_boxes = torch.tensor(particle_boxes)
    return particles_new, particle_boxes, colors


# Use SAM to extract particles, similiar to CryoSegNet
# NOTE: in cv2, image is stored as (w,h)
# sam return bbox with (center x , center y, w ,h)


def extract_sam(
    img: torch.tensor,
    origin_img: torch.tensor,
    meta,
    diameter: int,
    separation: float,
    pixelSize: float,
    artifects_threshold=0.01,
    iou_threshold=0.94,
    cut_edge=False,
    flip_Y=True,
):
    # img (1,1,1024,1024) with sigmoid (without normalize!)
    # sam mask generator read in image (h, w, c)
    _, _, scale_h, scale_w = img.shape

    h, w = meta["oh"], meta["ow"]
    scale = meta["scale"].numpy()

    # NOTE filter artifects
    img_filter_mask = torch.where(img > artifects_threshold, 1, 0)
    img = img * img_filter_mask
    # que
    # img = img[
    #     :, :, meta["pad_top"] : meta["target_size"] - meta["pad_bottom"], meta["pad_left"] :meta["target_size"] - meta["pad_right"]
    # ]

    predicted_mask = torch.cat((img, img, img), dim=1)
    predicted_mask = predicted_mask.squeeze(0).permute(1, 2, 0)

    sam_input = predicted_mask.cpu().numpy()

    masks = mask_generator.generate(sam_input)

    bboxes = []

    for i in range(0, len(masks)):
        if masks[i]["predicted_iou"] > iou_threshold:
            # box (center x ,center y , w, h)
            box = masks[i]["bbox"]
            bboxes.append(box)

    # second pass
    picking_method = "parseek"
    match picking_method:
        # CryoSegNet: second pass with mode diameter (-1/15 d ~ +1/10 d)
        case "cryoseg":
            particles = []
            if len(bboxes) > 0:
                mode_w = st.mode([box[2] for box in bboxes])  # w
                mode_h = st.mode([box[3] for box in bboxes])  # h
                mean_d_ = (mode_w + mode_h) / 2.0
                r_ = int(mean_d_ // 2)

                # th = 1/10 d_
                th = r_ * 0.20
                th_w = mode_w * 0.1
                th_h = mode_h * 0.1
                for b in bboxes:
                    if (
                        b[2] < mode_w + th_w
                        and b[2] > mode_w - th_w / 3
                        and b[3] < mode_h + th_h
                        and b[3] > mode_h - th_h / 3
                    ):
                        particles.extend(
                            [b[0] + (b[2] / 2.0), b[1] + (b[3] / 2.0)])
            else:
                particles = pd.DataFrame(columns=["x", "y"])

        # In production: second pass with mode diameter (-1/3 d ~ +1/3 d)
        case "GMM":
            particles = []
            if len(bboxes) > 0:
                mode_w = st.mode([box[2] for box in bboxes])  # w
                mode_h = st.mode([box[3] for box in bboxes])  # h
                # print(mode_w, mode_h)
                mean_d_ = np.sqrt(
                    (mode_w * meta["scale"]) ** 2 +
                    (mode_h * meta["scale"]) ** 2
                )
                r_ = int(mean_d_ // 2)
                # th = 1/10 d_
                th = r_ * 0.20
                for b in bboxes:
                    if (
                        b[2] < mode_w + th
                        and b[2] > mode_w - th / 3
                        and b[3] < mode_h + th
                        and b[3] > mode_h - th / 3
                    ):
                        particles.extend(
                            [b[0] + (b[2] / 2.0), b[1] + (b[3] / 2.0)])
            else:
                particles = pd.DataFrame(columns=["x", "y"])
        # Use all particles
        case "NF":
            particles = []
            if len(bboxes) > 0:
                for b in bboxes:
                    particles.extend(
                        [b[0] + (b[2] / 2.0), b[1] + (b[3] / 2.0)])
            else:
                particles = pd.DataFrame(columns=["x", "y"])
        # ParSeek use (0.7d ~ 1.3d)
        case "parseek":
            particles = []
            if len(bboxes) > 0:
                mode_w = st.mode([box[2] for box in bboxes])  # w
                mode_h = st.mode([box[3] for box in bboxes])  # h
                mean_d_ = (mode_w + mode_h) // 2
                th = mean_d_ * 0.3
                for b in bboxes:
                    if (
                        b[2] < mean_d_ + th
                        and b[2] > mean_d_ - th
                        and b[3] < mean_d_ + th
                        and b[3] > mean_d_ - th
                    ):
                        particles.extend(
                            [b[0] + (b[2] / 2.0), b[1] + (b[3] / 2.0)])
            else:
                particles = pd.DataFrame(columns=["x", "y"])

    n_particles = len(particles) // 2
    particles = np.array(particles)
    particles = pd.DataFrame(particles.reshape(
        n_particles, 2), columns=["x", "y"])

    if flip_Y:
        particles.y = img.shape[-2] - particles.y

    particles.x = particles.x * (scale)
    particles.y = particles.y * (scale)

    particles.x = (particles.x).astype(np.int32)
    particles.y = (particles.y).astype(np.int32)

    # filtering particles near the edge
    # default cutoff use diameter // 2
    # for EMPIAR-10017, use a larger diamter
    if n_particles > 0:
        if cut_edge:
            # for empiar 10017
            particles_new = particles[
                ~particles.apply(
                    check_on_edge_pandas, axis=1, h=w, w=h, r=diameter // 2 + diameter
                )
            ]
        else:
            particles_new = particles[
                ~particles.apply(
                    check_on_edge_pandas, axis=1, h=w, w=h, r=diameter // 2
                )
            ]
    else:
        particles_new = particles

    # draw particle boxes in red
    particle_boxes = []
    color = "red"
    colors = []
    for index, row in particles_new.iterrows():
        colors.append(color)
        thisbox = boxes(
            int(row.x / meta["scale"]),
            int(row.y / meta["scale"]),
            int(diameter / meta["scale"]),
        )
        particle_boxes.append(thisbox)
        colors.append(color)

    particle_boxes = torch.tensor(particle_boxes)
    return particles_new, particle_boxes, colors


# write
def write_cryosparc_pick_star(
    particles, micrograph_file_path, star_file, star_file_path, relion_metadata, **opt
):
    relion_version = ["#", "version", "30001"]

    relion_optics = [
        "_rlnVoltage",
        "_rlnImagePixelSize",
        "_rlnSphericalAberration",
        "_rlnAmplitudeContrast",
        "_rlnOpticsGroup",
        "_rlnImageSize",
        "_rlnImageDimensionality",
        "_rlnOpticsGroupName",
    ]

    relion_optics_data = pd.DataFrame(dict(relion_metadata), index=[0])

    relion_data_columns = ["_rlnMicrographName",
                           "_rlnCoordinateX", "_rlnCoordinateY"]

    relion_star_metadata = [
        relion_version,
        relion_optics,
        relion_optics_data,
        relion_data_columns,
        "data_particles",
    ]

    relion_star_data = particles[["x", "y"]].copy().astype(int)
    relion_star_data.rename(columns={"x": "_rlnCoordinateX"}, inplace=True)
    relion_star_data.rename(columns={"y": "_rlnCoordinateY"}, inplace=True)
    relion_star_data["_rlnMicrographName"] = micrograph_file_path

    relion_star_data = relion_star_data[
        ["_rlnMicrographName", "_rlnCoordinateX", "_rlnCoordinateY"]
    ]
    if particles.shape[0] > 0:
        fileparser.writestar(
            relion_star_data, relion_star_metadata, star_file_path)


# write _pick.star for importing by relion
def write_relion_pick_star(
    particles, micrograph_file_path, star_file, star_file_path, relion_metadata, **opt
):
    relion_version = ["#", "version", "30001"]

    relion_optics = [
        "_rlnVoltage",
        "_rlnImagePixelSize",
        "_rlnSphericalAberration",
        "_rlnAmplitudeContrast",
        "_rlnOpticsGroup",
        "_rlnImageSize",
        "_rlnImageDimensionality",
        "_rlnOpticsGroupName",
    ]

    relion_optics_data = pd.DataFrame(dict(relion_metadata), index=[0])

    relion_data_columns = ["_rlnCoordinateX", "_rlnCoordinateY"]

    relion_star_metadata = [
        relion_version,
        [],  # relion_optics,
        [],  # relion_optics_data,
        relion_data_columns,
        "data_",
    ]

    relion_star_data = particles[["x", "y"]].copy().astype(int)
    relion_star_data.rename(columns={"x": "_rlnCoordinateX"}, inplace=True)
    relion_star_data.rename(columns={"y": "_rlnCoordinateY"}, inplace=True)
    # relion_star_data["_rlnMicrographName"] = micrograph_file_path

    relion_star_data = relion_star_data[["_rlnCoordinateX", "_rlnCoordinateY"]]
    if particles.shape[0] > 0:
        fileparser.writestar(
            relion_star_data, relion_star_metadata, star_file_path, relegate=True
        )

# use for denoise


def process_patches(model, x: torch.tensor, patch_size, padding=128):
    """modified from topaz denoise_patches"""
    n, c, h, w = x.shape
    # print(n, c, h, w)
    y = torch.zeros_like(x)
    full_pad = torch.nn.functional.pad(
        x, (padding, padding, padding, padding), "reflect"
    )

    with torch.no_grad():
        for i in range(0, x.size(2), patch_size):
            for j in range(0, x.size(3), patch_size):
                if i >= 0 and i + patch_size <= x.size(2):
                    si = i
                    ei = i + patch_size
                else:
                    si = x.size(2) - patch_size
                    ei = x.size(2)

                if j >= 0 and j + patch_size <= x.size(3):
                    sj = j
                    ej = j + patch_size
                else:
                    sj = x.size(3) - patch_size
                    ej = x.size(3)

                xij = full_pad[
                    :,
                    :,
                    si + padding - padding: ei + padding + padding,
                    sj + padding - padding: ej + padding + padding,
                ]

                yij = model(xij)  # .squeeze(0)

                y[:, :, si:ei, sj:ej] = yij[
                    :, :, padding: padding + patch_size, padding: padding + patch_size
                ]
    y = y
    return y


def process_patches_unfold(model, x: torch.tensor, patch_size, padding=128):
    return 0


def diagonal_hard_cut(
    img1,
    img2,
):
    assert img1.shape == img2.shape
    n, c, h, w = img1.shape
    y_coords = torch.arange(h).float().view(-1, 1)
    x_coords = torch.arange(w).float().view(1, -1)

    mask = (y_coords / h + x_coords / w < 1.0).float()
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask * img1 + (1 - mask) * img2


# que
"""
def highpass_filter_gaussian(
    input_tensor: torch.Tensor,
    cutoff_frequency: float,
    pixel_size: float = 1.0
) -> torch.Tensor:
    import kornia.filters as filters
    sigma_pixels = cutoff_frequency / pixel_size / (2 * math.pi)

    kernel_size = int(2 * math.ceil(3 * sigma_pixels) + 1)
    kernel_size = min(kernel_size, min(input_tensor.shape[-2:]) // 4)

    low_freq = filters.gaussian_blur2d(
        input_tensor,
        kernel_size=(kernel_size, kernel_size),
        sigma=(sigma_pixels, sigma_pixels),
        border_type='reflect'
    )
    high_freq = input_tensor - low_freq
    return high_freq
"""

# que
"""
def lowpass_filter_gaussian(
    input_tensor: torch.Tensor,
    cutoff_frequency: float,
    pixel_size: float = 1.0
) -> torch.Tensor:
    sigma_pixels = cutoff_frequency / pixel_size / (2 * math.pi)

    kernel_size = int(2 * math.ceil(3 * sigma_pixels) + 1)
    kernel_size = min(kernel_size, min(input_tensor.shape[-2:]) // 4)

    low_freq = filters.gaussian_blur2d(
        input_tensor,
        kernel_size=(kernel_size, kernel_size),
        sigma=(sigma_pixels, sigma_pixels),
        border_type='reflect'
    )
    return low_freq
"""

# que

"""
def cryo_em_bandpass_filter(
    input_tensor: torch.Tensor,
    remove_high_freq: bool = True,
    remove_low_freq: bool = True,
) -> torch.Tensor:
    result = input_tensor.clone()

    if remove_high_freq:
        result = filters.gaussian_blur2d(
            result, kernel_size=(7, 7), sigma=(1.2, 1.2), border_type="reflect"
        )

    if remove_low_freq:
        low_freq_background = filters.gaussian_blur2d(
            input_tensor,
            kernel_size=(51, 51),
            sigma=(12.0, 12.0),
            border_type="reflect",
        )
        result = result - low_freq_background * 0.3

    return result
"""


def kornia_scale_4x_cycle(input_tensor: torch.Tensor):
    B, C, H, W = input_tensor.shape

    downsampled = kgt.resize(
        input_tensor,
        size=(H // 4, W // 4),
        interpolation='bilinear',
        align_corners=False,
        antialias=True
    )
    upsampled = kgt.resize(
        downsampled,
        size=(H, W),
        interpolation='bicubic',
        align_corners=False,
        antialias=True
    )
    return upsampled


def apps_denoise(opt):
    SAFE_WOKERS = 4
    save_executer = ThreadPoolExecutor(max_workers=SAFE_WOKERS)
    save_futuers = []

    def async_save_images(img, save_path, normalize=False):
        try:
            save_image(img, save_path, normalize=False)
        except Exception as e:
            print(f"Faile to save the image {save_path}: {e}")

    denoise_Dataset_opt = opt["inference_datasets"]
    denoise_Dataset = getattr(__import__("dataset"),
                              denoise_Dataset_opt["dataset"])

    # set target_size to -1 ,full size micrograph is used
    denoise_Dataset_opt["args"]["target_size"] = -1

    denoise_set = build(denoise_Dataset, denoise_Dataset_opt["args"])
    denoise_loader = DataLoader(
        denoise_set,
        batch_size=opt["batch_size"],
        shuffle=False,
        num_workers=28,
        drop_last=False,
    )

    denoise_output_root = os.path.join(
        opt["log_dir"]
    )

    net = build(
        getattr(__import__("network"),
                opt["networks"]["type"]), opt["networks"]["args"]
    )
    state_dict = torch.load(opt["networks"]["path"])
    net.load_state_dict(state_dict)

    device = opt["device"]
    device = torch.device(device)

    if not os.path.exists(denoise_output_root):
        os.mkdir(denoise_output_root)

    net.to(device)
    net.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda"):
            for step, (data, name, meta, _) in enumerate(tqdm.tqdm(denoise_loader)):
                if torch.isnan(data).any():
                    continue

                data_og = meta["std"].view(-1, 1, 1, 1) * data + meta["mean"].view(
                    -1, 1, 1, 1
                )
                # que
                # data_og = normalize(data)[0]
                # data_og = cryo_em_bandpass_filter(data_og,15.0,1.0)
                data_og = kornia_scale_4x_cycle(data_og)
                # data_og = lowpass_filter_gaussian(data_og,20.0,1.0 )
                data_og = normalize(data_og)[0]
                # size similar to train
                output = process_patches(
                    net, data.to(device), patch_size=640, padding=160
                )
                topaz = False
                if topaz:
                    topaz_mic = "/data/parsed2/aiparse_result/topaz_deniose/10017/" + \
                        name[0]+".jpg"
                    topaz_img = Image.open(topaz_mic)
                    topaz_img = np.asarray(topaz_img)
                    topaz_img = topaz_img.astype(np.float32)
                    topaz_img = topaz_img[np.newaxis, np.newaxis, :]
                    topaz_img = torch.Tensor(topaz_img)
                    topaz_img = normalize(topaz_img)[0]

                # clahe
                output = normalize(output)[0]
                # 2.0 or 3.0, not larger than 4.0
                output = equalize_clahe(
                    output, clip_limit=2.0, grid_size=(16, 16))
                output = normalize(output.contiguous())[0].to("cpu")

                psnr_value = psnr(normalize(output)[0], data_og, 1.0)

                # TODO remove test code
                # save denoised and origin images
                for i in range(data.shape[0]):
                    if topaz:

                        img_to_save = diagonal_hard_cut(
                            output[i: i + 1], topaz_img)
                        save_path = os.path.join(
                            denoise_output_root, name[i] + ".jpg")

                        future = save_executer.submit(
                            async_save_images,
                            img_to_save,
                            save_path,
                            normalize=False
                        )
                        save_futuers.append(future)

                    else:

                        img_to_save = diagonal_hard_cut(
                            output[i: i + 1], data_og[i: i + 1])
                        save_path = os.path.join(
                            denoise_output_root, name[i] + ".jpg")
                        future = save_executer.submit(
                            async_save_images,
                            img_to_save,
                            save_path,
                            normalize=False
                        )
                        save_futuers.append(future)

                # calculate psnr
                # TODO do psnr only on particles?
                # p = psnr.peak_signal_noise_ratio(
                #    normalize(output)[0].to("cuda"), data_og.to("cuda")
                # )

                log(opt["log_file"], "%s:%.2f\n" % (name[0], psnr_value))
                del data_og, output
                torch.cuda.empty_cache()
    wait(save_futuers, return_when=ALL_COMPLETED)
    save_executer.shutdown(wait=True)
    return 0


def apps_pick(opt):
    denoise_pick_opt = opt["apps"]
    denoiseDataset = getattr(__import__("dataset"),
                             denoise_pick_opt["dataset"])
    denoise_set = build(denoiseDataset, denoise_pick_opt["args"])
    denoise_loader = DataLoader(
        denoise_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False
    )

    pick_config_opt = opt["pick"]

    pick_output_root = os.path.join(
        opt["log_dir"]
    )

    relion_metadata = opt["relion_metadata"]

    net_pick = build(
        getattr(__import__("network"), opt["pick_network"]["type"]),
        opt["pick_network"]["args"],
    )

    pick_mode = opt["pick_network"]["pick_mode"]

    pick_state_dict = torch.load(opt["pick_network"]["path"])
    net_pick.load_state_dict(pick_state_dict)

    device_gpu = opt["device_pick"]
    device_gpu = torch.device(device_gpu)
    if not os.path.exists(pick_output_root):
        os.mkdir(pick_output_root)

    summary(net_pick, input_size=(1, 1, 1024, 1024))

    net_pick.eval().to(device_gpu)

    d = pick_config_opt["diameter"]

    count = 0

    global_metric_Dict = {
        "TotalPicked": 0,
        "GT": 0,
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1": 0.0,
    }

    with torch.no_grad():
        for step, (data, name, meta, ppp_gt) in enumerate(tqdm.tqdm(denoise_loader)):

            _, _, h, w = data.shape

            with torch.cuda.amp.autocast():
                pred_mask = process_patches(
                    net_pick, data.to("cuda:0"), 1024, 128)

                # do sigmoid with net_pick output , no normalize
                pred_mask = torch.sigmoid(pred_mask)

            # locate particles with sam
            match pick_mode:
                case "sam":
                    particles, particles_box, colors = extract_sam(
                        pred_mask, data, meta, **pick_config_opt
                    )
                case "dilate":
                    particles, particles_box, colors = extract_dilate(
                        pred_mask, data, **pick_config_opt
                    )
                case _:
                    raise NotImplementedError(
                        f"{pick_mode} Picking method is not implemented!"
                    )

            # transform back to origin image size after picking
            tr_back = v2.Resize((meta["oh"], meta["ow"]))
            output_seg = tr_back((pred_mask.squeeze(0)))
            output_seg = output_seg.unsqueeze(0)
            # do normalize for display

            output_seg = output_seg.to("cpu")

            # write out particle star
            write_format = "relion"
            match write_format:
                case "cryosparc":
                    write_cryosparc_pick_star(
                        particles,
                        name[0] + ".jpg",
                        name[0] + ".star",
                        os.path.join(pick_output_root, name[0] + ".star"),
                        relion_metadata,
                    )
                case "relion":
                    write_relion_pick_star(
                        particles,
                        name[0] + ".mrc",
                        name[0] + "_pick.star",
                        os.path.join(pick_output_root,
                                     name[0] + "_pick.star"),
                        relion_metadata,
                    )

            # compare pick result with cryoppp ground truth
            if len(ppp_gt) > 0:
                if os.path.exists(ppp_gt[0]):
                    ppp_gt_csv = pd.read_csv(ppp_gt[0])
                    pick_result, metric_particles_boxes, metric_colors, metric_dict = (
                        pick_metrics(
                            output_seg,
                            name[0],
                            particles,
                            ppp_gt_csv,
                            diameter=d,
                            ratio=0.3,
                            invertx=False,
                            inverty=False,
                            swapxy=False,
                            flag=0,
                        )
                    )
                    global_metric_Dict["TotalPicked"] += metric_dict["TotalPicked"]
                    global_metric_Dict["GT"] += metric_dict["GT"]
                    global_metric_Dict["TP"] += metric_dict["TP"]
                    global_metric_Dict["FP"] += metric_dict["FP"]
                    global_metric_Dict["FN"] += metric_dict["FN"]

                    # save pick on seg micrograph for check
                    res2 = draw_bounding_boxes_wrapper(
                        output_seg, metric_particles_boxes, metric_colors
                    )
                    save_image(
                        res2,
                        os.path.join(pick_output_root,
                                     name[0] + "_metric" + ".jpg"),
                        normalize=False,
                    )
                else:
                    print("groundTruth csv for %s is Not Found!" % ppp_gt[0])

    # summary of global_metric_Dict
    if len(ppp_gt) > 0:
        if global_metric_Dict["TotalPicked"] != 0:
            global_metric_Dict["Precision"] = (
                global_metric_Dict["TP"] / global_metric_Dict["TotalPicked"]
            )
        else:
            global_metric_Dict["Precision"] = 0.0

        global_metric_Dict["Recall"] = (
            global_metric_Dict["TP"] / global_metric_Dict["GT"]
        )
        global_metric_Dict["F1-score"] = (
            2
            * global_metric_Dict["Precision"]
            * global_metric_Dict["Recall"]
            / (global_metric_Dict["Precision"] + global_metric_Dict["Recall"])
        )

        log(opt["log_file"], ",Summary of Project\n")
        log(
            opt["log_file"],
            ",TotalPicked:%d,GroundTruth:%d,TruePicked:%d,Precision:%.2f,Recall:%.2f,F1-score:%.2f\n"
            % (
                global_metric_Dict["TotalPicked"],
                global_metric_Dict["GT"],
                global_metric_Dict["TP"],
                global_metric_Dict["Precision"],
                global_metric_Dict["Recall"],
                global_metric_Dict["F1-score"],
            ),
        )

    return 0


def apps_benchmark(opt):
    """
    compare topaz/cryoseg result star with cryoPPP ground truth csv
    """
    run_opt = opt["apps"]
    topaz_dataset = getattr(__import__("dataset"), run_opt["dataset"])
    topaz_set = build(topaz_dataset, run_opt["args"])
    topaz_loader = DataLoader(
        topaz_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False
    )
    # star_handle_method: topaz | cryoseg_metric
    star_handle_method = run_opt["args"]["ppp_format"]
    pick_config_opt = opt["pick"]

    aiparse_output_root = os.path.join(
        opt["aiparse_output_path"], opt["apps"]["args"]["id"]
    )

    if not os.path.exists(aiparse_output_root):
        os.mkdir(aiparse_output_root)

    d = pick_config_opt["diameter"]

    global_metric_Dict = {
        "TotalPicked": 0,
        "GT": 0,
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1": 0.0,
    }

    for step, (data, name, data_mean, data_std, ppp_gt, topaz_star) in enumerate(
        tqdm.tqdm(topaz_loader)
    ):
        if os.path.exists(ppp_gt[0]) and os.path.exists(topaz_star[0]):
            ppp_gt_csv = pd.read_csv(ppp_gt[0])
            match star_handle_method:
                case "topaz":
                    topaz_particles = fileparser.getparticles_dummyoptics(
                        topaz_star[0]
                    )[0]
                    topaz_particles["_rlnAutopickFigureOfMerit"] = topaz_particles[
                        "_rlnAutopickFigureOfMerit"
                    ].map(lambda x: float(x))
                    topaz_particles = topaz_particles[
                        topaz_particles["_rlnAutopickFigureOfMerit"] > 2.0
                    ]
                case "cryoseg_metric":
                    topaz_particles = fileparser.getparticles(topaz_star[0])[0]

            topaz_particles_pd = topaz_particles.loc[
                :, ["_rlnCoordinateX", "_rlnCoordinateY"]
            ]
            topaz_particles_pd.rename(
                columns={"_rlnCoordinateX": "x", "_rlnCoordinateY": "y"}, inplace=True
            )

            img = normalize(data * data_std + data_mean)[0]
            match star_handle_method:
                case "topaz":
                    pick_result, metric_particles_boxes, metric_colors, metric_dict = (
                        pick_metrics(
                            normalize(data * data_std + data_mean)[0],
                            name[0],
                            topaz_particles_pd,
                            ppp_gt_csv,
                            diameter=d,
                            ratio=0.3,
                            invertx=False,
                            inverty=True,
                            swapxy=False,
                            flag=1,
                        )
                    )
                case "cryoseg_metric":
                    pick_result, metric_particles_boxes, metric_colors, metric_dict = (
                        pick_metrics(
                            normalize(data * data_std + data_mean)[0],
                            name[0],
                            topaz_particles_pd,
                            ppp_gt_csv,
                            diameter=d,
                            ratio=0.3,
                            invertx=False,
                            inverty=True,
                            swapxy=False,
                            flag=1,
                        )
                    )

            global_metric_Dict["TotalPicked"] += metric_dict["TotalPicked"]
            global_metric_Dict["GT"] += metric_dict["GT"]
            global_metric_Dict["TP"] += metric_dict["TP"]
            global_metric_Dict["FP"] += metric_dict["FP"]
            global_metric_Dict["FN"] += metric_dict["FN"]
            res = draw_bounding_boxes_wrapper(
                img, metric_particles_boxes, metric_colors
            )
            save_image(
                res,
                os.path.join(aiparse_output_root,
                             name[0] + "_metric" + ".jpg"),
                normalize=False,
            )
        elif not os.path.exists(ppp_gt[0]):
            print("groundTruth csv for %s is Not Found!" % ppp_gt[0])
        else:
            print("topaz pick result star for %s is Not Found!" % topaz_star)

    # summary of global_metric_Dict
    if global_metric_Dict["TotalPicked"] != 0:
        global_metric_Dict["Precision"] = (
            global_metric_Dict["TP"] / global_metric_Dict["TotalPicked"]
        )
    else:
        global_metric_Dict["Precision"] = 0.0
    global_metric_Dict["Recall"] = global_metric_Dict["TP"] / \
        global_metric_Dict["GT"]
    global_metric_Dict["F1-score"] = (
        2
        * global_metric_Dict["Precision"]
        * global_metric_Dict["Recall"]
        / (global_metric_Dict["Precision"] + global_metric_Dict["Recall"])
    )

    log(opt["log_file"], ",Summary of Project\n")
    log(
        opt["log_file"],
        ",TotalPicked:%d,GroundTruth:%d,TruePicked:%d,Precision:%.2f,Recall:%.2f,F1-score:%.2f\n"
        % (
            global_metric_Dict["TotalPicked"],
            global_metric_Dict["GT"],
            global_metric_Dict["TP"],
            global_metric_Dict["Precision"],
            global_metric_Dict["Recall"],
            global_metric_Dict["F1-score"],
        ),
    )

    return 0


def pick_once(opt):
    debug_flag = opt["debug"]
    # pick_config_opt = opt["pick"]
    aiparse_output_root = opt["aiparse_output_path"]

    net_pick = build(
        getattr(__import__("network"), opt["pick_network"]["type"]),
        opt["pick_network"]["args"],
    )

    pick_state_dict = torch.load(opt["pick_network"]["path"])
    net_pick.load_state_dict(pick_state_dict)

    device_pick = opt["device_pick"]
    device_2 = torch.device(device_pick)
    net_pick.eval().to(device_2)

    match opt["apps"]["args"]["ppp_format"]:
        case "ppp":
            image_dir = os.path.join(
                opt["apps"]["args"]["root_dir"],
                opt["apps"]["args"]["id"],
                "micrographs",
            )  # , opt["image_format"])
        case "png":
            image_dir = os.path.join(
                opt["apps"]["args"]["root_dir"], opt["apps"]["args"]["id"], "bnn"
            )  # , opt["image_format"])
    image_path = os.path.join(image_dir, opt["apps"]["args"]["image"])
    # name = opt["apps"]["args"]["image"].split(".mrc")[0]
    suffix = "_" + opt["apps"]["args"]["suffix"]
    colors = opt["apps"]["args"]["colors"]
    print(image_path)

    match opt["apps"]["args"]["ppp_format"]:
        case "mrc" | "ppp":
            img = mrcfile.open(image_path, permissive=True)
            img = img.data
            img = img.astype(np.float32)
            name = opt["apps"]["args"]["image"].split(".mrc")[0]
        case "jpeg" | "jpg":
            name = opt["apps"]["args"]["image"].split(".jpg")[0]
            img = Image.open(image_path)
            img = np.asarray(img)
            img = img.astype(np.float32)
            img = img / 255.0
        case "png":
            name = opt["apps"]["args"]["image"].split(".jpg")[0]
            img = Image.open(image_path)
            img = np.asarray(img)
            print(img.shape)
            img = img[:, :, 0]
            img = img.astype(np.float32)
            img = img / 255.0

    print(name)
    data = zscore(img)[0]
    data = torch.tensor(data)
    data = data.unsqueeze(0)
    tr = v2.Resize((1024, 1024))
    resize_img = tr((data))
    resize_img = resize_img.unsqueeze(0)
    print(resize_img.shape)

    with torch.no_grad():
        resize_output = net_pick(resize_img.cuda())

    resize_output = torch.sigmoid(resize_output)
    from kornia.filters import box_blur, gaussian_blur2d
    from kornia.enhance import equalize_clahe

    # resize_img  = box_blur(resize_img,(3,3))
    resize_img = gaussian_blur2d(resize_img, (5, 5), (1.5, 1.5))
    resize_img = normalize(resize_img)[0]
    print(resize_img.shape)
    resize_img = equalize_clahe(resize_img, clip_limit=2.0, grid_size=(16, 16))
    print(resize_img.shape)

    res = draw_segmentation_masks_wrapper(
        resize_img, resize_output, 0.6, colors=colors)
    save_image(
        normalize(resize_img)[0], os.path.join(
            aiparse_output_root, name + ".png")
    )
    save_image(
        res, os.path.join(aiparse_output_root, name +
                          suffix + "_segmask" + ".png")
    )
    save_image(
        resize_output, os.path.join(
            aiparse_output_root, name + suffix + "_seg.png")
    )

    return 0


def production_pick(opt):
    debug_flag = opt["debug"]
    denoise_pick_opt = opt["apps"]
    denoiseDataset = getattr(__import__("dataset"),
                             denoise_pick_opt["dataset"])
    denoise_set = build(denoiseDataset, denoise_pick_opt["args"])
    denoise_loader = DataLoader(
        denoise_set, batch_size=1, shuffle=False, num_workers=4, drop_last=False
    )

    """
    net_mode: cryoseg | parse | parse_wo | parse_bnn | patch | patch_wo | patch_bnn
    cryoseg: gunet trained with real cryoppp
    parse: gunet trained with simulate cryoppp (with cryoseg denoise) (default)
    parse_wo: gunet trained with simulate cryoppp (without any preprocessing)
    parse_bnn: gunet trained with simulate cryoppp (with bnn denoise)

    patch: gunet trained with simulate cryoppp (with cryoseg denoise) (default)
    patch_wo: gunet trained with simulate cryoppp (without any preprocessing)
    patch: gunet trained with simulate cryoppp (with bnn denoise)
    """

    preprocess_mode = denoise_pick_opt["args"]["ppp_format"]

    pick_config_opt = opt["pick"]

    aiparse_output_root = os.path.join(
        # opt["aiparse_output_path"], opt["apps"]["args"]["id"]
        opt["log_dir"]
    )

    relion_metadata = opt["relion_metadata"]

    net_denoise = build(
        getattr(__import__("network"), opt["denoise_network"]["type"]),
        opt["denoise_network"]["args"],
    )

    denoise_state_dict = torch.load(opt["denoise_network"]["path"])
    net_denoise.load_state_dict(denoise_state_dict)

    net_pick = build(
        getattr(__import__("network"), opt["pick_network"]["type"]),
        opt["pick_network"]["args"],
    )

    net_mode = opt["pick_network"]["net_mode"]
    pick_mode = opt["pick_network"]["pick_mode"]

    pick_state_dict = torch.load(opt["pick_network"]["path"])
    net_pick.load_state_dict(pick_state_dict)

    denoise_device = opt["device_denoise"]
    denoise_device = torch.device(denoise_device)
    seg_device = opt["device_pick"]
    seg_device = torch.device(seg_device)
    sam_device = opt["sam_device"]
    sam_device = torch.device(sam_device)

    if not os.path.exists(aiparse_output_root):
        os.mkdir(aiparse_output_root)

    net_denoise.eval().to(denoise_device)

    # 768 + 2*128 = 1024
    summary(net_pick, input_size=(1, 1, 1024, 1024))
    net_pick.eval().to(seg_device)
    # 1024 + 128*2 = 1280

    d = pick_config_opt["diameter"]

    count = 0

    global_metric_Dict = {
        "TotalPicked": 0,
        "GT": 0,
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1": 0.0,
    }
    # main -> denoise -> seg -> sam -> postprocess
    queue_main = queue.Queue(maxsize=4)
    queue_denoise = queue.Queue(maxsize=4)
    queue_seg = queue.Queue(maxsize=2)
    queue_sam = queue.Queue(maxsize=1)
    queue_postprocess = queue.Queue(maxsize=4)

    def stage_denoise(x):
        while True:
            data, name, data_mean, data_std, ppp_gt = queue_main.get()
            if data is None:
                queue_denoise.put(
                    (None, data, name, data_mean, data_std, ppp_gt))
            else:
                if denoise_flag:
                    output_denoise = process_patches(
                        net_denoise, data.to(denoise_device), 640, padding=160
                    )
                if clahe_flag:
                    output_denoise = (output_denoise * data_std) + data_mean
                    output_denoise = equalize_clahe(
                        output_denoise, clip_limit=2.0, grid_size=(16, 16)
                    )
                    output_denoise = normalize(output_denoise)[0]

                queue_denoise.put(
                    (
                        output_denoise.to(seg_device, non_blocking=True),
                        data,
                        name,
                        data_mean,
                        data_std,
                        ppp_gt,
                    )
                )

    def stage_seg(x):
        while True:
            output_denoise, data, name, data_mean, data_std, ppp_gt = (
                queue_denoise.get()
            )
            if output_denoise is None:
                queue_seg.put((None, data, name, data_mean, data_std, ppp_gt))
            else:
                match net_mode:
                    case "cryoseg" | "parse" | "parse_wo" | "parse_bnn" | "ppp":
                        tr = v2.Resize((1024, 1024))
                        output_denoise = output_denoise.to("cuda")
                        resize_img = tr((output_denoise.squeeze(0)))
                        resize_img = resize_img.unsqueeze(0)

                        # in full mode , net_pick is training with image size of 1024
                        with torch.cuda.stream(torch.cuda.Stream(device=seg_device)):
                            resize_output = net_pick(resize_img)
                            # do sigmoid with net_pick output , no normalize
                            resize_output = torch.sigmoid(resize_output)

                        queue_seg.put(
                            (
                                resize_output.to(
                                    sam_device, non_blocking=True),
                                data,
                                name,
                                data_mean,
                                data_std,
                                ppp_gt,
                            )
                        )

    def stage_sam(x):
        while True:
            resize_output, data, name, data_mean, data_std, ppp_gt = queue_seg.get()
            if resize_output is None:
                queue_sam.put((None, data, name, data_mean, data_std, ppp_gt))
            else:
                match pick_mode:
                    case "sam":
                        particles, particles_box, colors = extract_sam(
                            resize_output, data, **pick_config_opt
                        )
                    case "dilate":
                        particles, particles_box, colors = extract_dilate(
                            resize_output, data, **pick_config_opt
                        )
                    case _:
                        raise NotImplementedError(
                            f"{pick_mode} Picking method is not implemented!"
                        )
                tr_back = v2.Resize((h, w))
                output_seg = tr_back((resize_output.squeeze(0)))
                output_seg = output_seg.unsqueeze(0)
                queue_sam.put(output_seg, None, data, name,
                              data_mean, data_std, ppp_gt)

    def stage_postprocess(x):
        while True:
            match write_format:
                case "cryosparc":
                    pass
                case "relion":
                    pass

    with torch.no_grad():
        for step, (data, name, data_mean, data_std, ppp_gt) in enumerate(
            tqdm.tqdm(denoise_loader)
        ):

            _, _, h, w = data.shape

            match preprocess_mode:
                case "cryoseg_bnn":
                    denoise_flag = True
                    clahe_flag = False
                case "cryoseg_bnc":
                    denoise_flag = True
                    clahe_flag = True
                case _:
                    denoise_flag = False
                    clahe_flag = False

            output_denoise = data

            # do bnn denoise
            if denoise_flag:
                output_denoise = process_patches(
                    net_denoise, output_denoise.to(denoise_device), 640, padding=160
                )

            # que
            # transfer image to cpu
            # output_denoise = output_denoise.to("cpu")
            # match preprocess_mode:
            #     case "ppp":
            #         output_denoise = output_denoise
            #     case _:
            #         output_denoise = (output_denoise * data_std) + data_mean
            #         output_denoise = zscore(output_denoise)[0]

            # do clahe with bnn output on gpu
            if clahe_flag:
                output_denoise = (output_denoise * data_std) + data_mean
                output_denoise = equalize_clahe(
                    output_denoise, clip_limit=2.0, grid_size=(16, 16)
                )
                output_denoise = normalize(output_denoise)[0]

            # pick
            # transfer image to cpu

            # que
            # for model trained with normlized input , CS_RT
            # for cryoseg input
            # output_denoise = data*data_std + data_mean
            # output_denoise = normalize(output_denoise)[0]
            # net_mode = "patch"

            match net_mode:
                case "cryoseg" | "parse" | "parse_wo" | "parse_bnn" | "ppp":
                    tr = v2.Resize((1024, 1024))
                    output_denoise = output_denoise.to("cuda")
                    resize_img = tr((output_denoise.squeeze(0)))
                    resize_img = resize_img.unsqueeze(0)

                    # in full mode , net_pick is training with image size of 1024
                    resize_output = net_pick(resize_img)
                    # do sigmoid with net_pick output , no normalize
                    resize_output = torch.sigmoid(resize_output)

                    # locate particles with sam
                    match pick_mode:
                        case "sam":
                            particles, particles_box, colors = extract_sam(
                                resize_output, data, **pick_config_opt
                            )
                        case "dilate":
                            particles, particles_box, colors = extract_dilate(
                                resize_output, data, **pick_config_opt
                            )
                        case _:
                            raise NotImplementedError(
                                f"{pick_mode} Picking method is not implemented!"
                            )
                    # transform back to origin image size after picking
                    tr_back = v2.Resize((h, w))
                    output_seg = tr_back((resize_output.squeeze(0)))
                    output_seg = output_seg.unsqueeze(0)
                    # do normalize for display
                    # output_seg = normalize(output_seg)[0]
                case "patch" | "patch_wo" | "patch_bnn":
                    output_denoise = output_denoise.to("cuda")
                    output_seg = process_patches(
                        net_pick, output_denoise, 1024, 128)
                    # sigmoid with net_pick output
                    output_seg = torch.sigmoid(output_seg)
                    tr = v2.Resize((1024, 1024))
                    resize_output = tr((output_seg))
                    particles, particles_box, colors = extract_sam(
                        resize_output, data, **pick_config_opt
                    )
                    # locate particles with trackpy
                    # particles, particles_boxes, colors = aiparse_pick_dilate(
                    #    output_seg, **pick_config_opt
                    # )
                case _:
                    raise NotImplementedError("Not known net_mode")

            output_seg = output_seg.to("cpu")

            # write out particle star
            write_format = "relion"
            match write_format:
                case "cryosparc":
                    write_cryosparc_pick_star(
                        particles,
                        name[0] + ".jpg",
                        name[0] + ".star",
                        os.path.join(aiparse_output_root, name[0] + ".star"),
                        relion_metadata,
                    )
                case "relion":
                    write_relion_pick_star(
                        particles,
                        name[0] + ".mrc",
                        name[0] + "_pick.star",
                        os.path.join(aiparse_output_root,
                                     name[0] + "_pick.star"),
                        relion_metadata,
                    )

            # compare pick result with cryoppp ground truth
            if len(ppp_gt) > 0:
                if os.path.exists(ppp_gt[0]):
                    ppp_gt_csv = pd.read_csv(ppp_gt[0])
                    pick_result, metric_particles_boxes, metric_colors, metric_dict = (
                        pick_metrics(
                            output_seg,
                            name[0],
                            particles,
                            ppp_gt_csv,
                            diameter=d,
                            ratio=0.3,
                            invertx=False,
                            inverty=False,
                            swapxy=False,
                            flag=0,
                        )
                    )
                    global_metric_Dict["TotalPicked"] += metric_dict["TotalPicked"]
                    global_metric_Dict["GT"] += metric_dict["GT"]
                    global_metric_Dict["TP"] += metric_dict["TP"]
                    global_metric_Dict["FP"] += metric_dict["FP"]
                    global_metric_Dict["FN"] += metric_dict["FN"]

                    # save pick on seg micrograph for check
                    res2 = draw_bounding_boxes_wrapper(
                        output_seg, metric_particles_boxes, metric_colors
                    )

                    # save pick on denoised micrograph for check
                    res3 = draw_bounding_boxes_wrapper(
                        output_denoise, metric_particles_boxes, metric_colors
                    )

                    save_image(
                        [res3, res2],
                        os.path.join(aiparse_output_root,
                                     name[0] + "_metric" + ".jpg"),
                        normalize=False,
                    )
                else:
                    print("groundTruth csv for %s is Not Found!" % ppp_gt[0])

    # summary of global_metric_Dict
    if len(ppp_gt) > 0:
        if global_metric_Dict["TotalPicked"] != 0:
            global_metric_Dict["Precision"] = (
                global_metric_Dict["TP"] / global_metric_Dict["TotalPicked"]
            )
        else:
            global_metric_Dict["Precision"] = 0.0

        global_metric_Dict["Recall"] = (
            global_metric_Dict["TP"] / global_metric_Dict["GT"]
        )
        global_metric_Dict["F1-score"] = (
            2
            * global_metric_Dict["Precision"]
            * global_metric_Dict["Recall"]
            / (global_metric_Dict["Precision"] + global_metric_Dict["Recall"])
        )

        log(opt["log_file"], ",Summary of Project\n")
        log(
            opt["log_file"],
            ",TotalPicked:%d,GroundTruth:%d,TruePicked:%d,Precision:%.2f,Recall:%.2f,F1-score:%.2f\n"
            % (
                global_metric_Dict["TotalPicked"],
                global_metric_Dict["GT"],
                global_metric_Dict["TP"],
                global_metric_Dict["Precision"],
                global_metric_Dict["Recall"],
                global_metric_Dict["F1-score"],
            ),
        )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise only")
    parser.add_argument("--config", type=str,
                        default="options/denoise_only.json")
    argspar = parser.parse_args()

    opt = parse(argspar.config)
    if os.path.exists(os.path.dirname(opt["log_dir"])):
        if not os.path.exists(opt["log_dir"]):
            os.mkdir(opt["log_dir"])

    recursive_log(opt["log_file"], opt)

    match opt["apps"]:
        case "parseek_pick":
            apps_pick(opt)
        case "parseek_denoise":
            apps_denoise(opt)
        case "parseeek_metric":
            apps_benchmark(opt)
        case "parseek_seg":
            pick_once(opt)
        case _:
            raise NotImplementedError
