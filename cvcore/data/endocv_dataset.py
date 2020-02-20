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


class EDDDataset(Dataset):
    def __init__(self, cfg, mode="train"):
        super(EDDDataset, self).__init__()
        assert mode in ("train", "valid", "test", "holdout")
        self.data_dir = cfg.DIRS.DATA
        self.mask0_dir = cfg.DIRS.MASK0
        self.mask1_dir = cfg.DIRS.MASK1
        self.img_size = (cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE)
        if mode in ("train", "valid"):
            self.df = pd.read_csv(f"./data/slices_mask/{mode}_fold{cfg.DATA.FOLD}.csv")
        if mode == "holdout":
            self.df = pd.read_csv(f"./data/slices_mask/holdout.csv")
        if mode == "train":
            self.aug = A.Compose([
                # A.OneOf([
                #     A.HorizontalFlip(p=1.),
                #     A.VerticalFlip(p=1.),
                # ]),
                A.HorizontalFlip(p=0.5),
                # A.ShiftScaleRotate(
                #         shift_limit=0.0625,
                #         scale_limit=0.1,
                #         rotate_limit=30,
                #         border_mode=cv2.BORDER_CONSTANT,
                #         value=0,
                #         p=1.),
                A.GridDistortion(
                    distort_limit=0.2,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=1.),
                A.OneOf([
                    A.IAAAdditiveGaussianNoise(),
                    A.GaussNoise(),
                ], p=0.5),
                # A.OneOf([
                #     A.MedianBlur(blur_limit=3, p=1.),
                #     A.Blur(blur_limit=3, p=1.),
                # ]),
                A.Normalize(),
                A.pytorch.ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
                A.Normalize(),
                A.pytorch.ToTensorV2()
            ])
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df['img'].values[idx]
        img = Image.open(os.path.join(self.data_dir, img_id + ".jpg")).convert("RGB")
        img = np.asarray(img, dtype=np.uint8)

        if self.mode != "test":
            mask = np.zeros((*img.shape[:-1], 2))
            mask[..., 0] = np.asarray(Image.open(os.path.join(self.mask0_dir, img_id + ".jpg")).convert('L'), dtype=np.uint8)
            mask[..., 1] = np.asarray(Image.open(os.path.join(self.mask1_dir, img_id + ".jpg")).convert('L'), dtype=np.uint8)
            mask = np.where(mask > 100, 255, 0)
            img = cv2.resize(img, self.img_size,
                interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.img_size,
                interpolation=cv2.INTER_NEAREST)
            data = self.aug(image=img, mask=mask)
            img, mask = data['image'], data['mask']
            mask = mask.permute(2, 0, 1).div(255.).float()
            return img, mask
        else:
            img = cv2.resize(img, self.img_size,
                interpolation=cv2.INTER_LINEAR)
            img = self.aug(image=img)['image']
            return img