from collections import OrderedDict
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import gc
import numpy as np
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations import pytorch
# from .auto_augment import Invert, RandAugment, AugmentAndMix


# HEIGHT = 512
# WIDTH = 512


def train_aug():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
    ])


def normalize():
    # return A.Normalize(mean=(0.485, 0.456, 0.406, 0),
    #     std=(0.229, 0.224, 0.225, 1))
    return A.Normalize(mean=(0, 0, 0, 0),
        std=(1, 1, 1, 1))


class AgVisDataset(Dataset):
    def __init__(self, cfg, mode="train"):
        super(AgVisDataset, self).__init__()
        if mode == "train":
            self.data_dir = cfg.DIRS.TRAIN
        elif mode == "valid":
            self.data_dir = cfg.DIRS.VALID
        else:
            self.data_dir = cfg.DIRS.TEST

        rgb_images_dir = os.path.join(self.data_dir, "images/rgb")
        self.rgb_images = [os.path.join(rgb_images_dir, f)
            for f in os.listdir(rgb_images_dir)]
        self.labels_to_indexes = {
            "cloud_shadow": 0,
            # "double_plant": 1,
            # "planter_skip": 2,
            "standing_water": 3,
            "waterway": 4,
            "weed_cluster": 5
        }
        if not mode != "test":
            pixel_stat = pd.read_csv(self.data_dir, f"{mode}_pixel_stat.csv")
            removed_img_ids = pixel_stat[(pixel_stat["cloud_shadow"]==0)&
                (pixel_stat["standing_water"]==0)&(pixel_stat["waterway"]==0)&
                (pixel_stat["weed_cluster"]==0)]["img_id"].values
            self.rgb_images = [os.path.join(rgb_images_dir, f)
                for f in os.listdir(rgb_images_dir)
                if f.split(".")[-1] not in removed_img_ids]
        self.mode = mode

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_fpath = self.rgb_images[idx]
        nir_fpath = rgb_fpath.replace("rgb", "nir")
        image = np.concatenate((
            cv2.imread(rgb_fpath, cv2.COLOR_BGR2RGB),
            cv2.imread(nir_fpath, cv2.IMREAD_GRAYSCALE)[..., np.newaxis],
        ), -1)

        if self.mode != "test":
            mask_fpaths = [rgb_fpath.replace("images/rgb", f"labels/{label}").replace("jpg", "png")
                for label in self.labels_to_indexes.keys()]
            mask = np.concatenate([
                cv2.imread(fp, cv2.IMREAD_GRAYSCALE)[..., np.newaxis] for fp in mask_fpaths], -1)
            boundary_fpath = rgb_fpath.replace("images/rgb", "boundaries").replace("jpg", "png")
            boundary = cv2.imread(boundary_fpath, cv2.IMREAD_GRAYSCALE)
        if self.mode == "train":
            data = train_aug()(image=image, masks=[mask, boundary])
            image, masks = data["image"], data["masks"]
            mask, boundary = masks

        image = normalize()(image=image)["image"]
        ndis = []
        for i in range(2):
            ndi = (image[..., -1] - image[..., i] + 1e-6) / (image[..., -1] + image[..., i] + 1e-6)
            ndis.append(ndi[..., np.newaxis])
        ndis = np.concatenate(ndis, -1)
        image = np.concatenate((image, ndis), -1)
        image = torch.from_numpy(image.transpose((2, 0, 1)))

        if self.mode == "test":
            img_id = rgb_fpath.split("/")[-1].replace(".jpg", "")
            return image, img_id
        else:
            mask = torch.from_numpy(mask.transpose((2, 0, 1))).div(255.).float()
            boundary = torch.from_numpy(boundary).div(255.).float()
            boundary = torch.stack([boundary] * len(mask))
            return image, mask, boundary