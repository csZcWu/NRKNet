import os.path
import time

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils import data
from config import data_offset


class Dataset(data.Dataset):
    def __init__(self, img_list_file, crop_size=(256, 256)):
        super(type(self), self).__init__()

        self.crop_size = crop_size
        st = time.time()
        print('loading data')
        f = open(img_list_file, 'r')
        lines = f.readlines()
        f.close()
        self.train_img_list = []
        self.train_gt_list = []
        for line in lines:
            img_name, gt_name = line[:-1].split(',')
            self.train_img_list.append(torch.from_numpy(np.array(cv2.imread(os.path.join(data_offset,img_name))).transpose((2, 0, 1))))
            self.train_gt_list.append(torch.from_numpy(np.array(cv2.imread(os.path.join(data_offset,gt_name))).transpose((2, 0, 1))))
        print('loading data finished', time.time() - st)

    def __len__(self):
        return len(self.train_gt_list)

    def data_argumentation(self, img, crop_left, crop_top, hf, vf, rot):
        img = img[:, crop_top:crop_top + self.crop_size[1], crop_left:crop_left + self.crop_size[0]]
        if hf:
            img = F.hflip(img)
        if vf:
            img = F.vflip(img)
        img = torch.rot90(img, rot, [1, 2])
        return img

    def __getitem__(self, idx):

        clear_img = self.train_gt_list[idx]
        blurry_img = self.train_img_list[idx]
        _, h, w = clear_img.shape
        crop_left = int(np.floor(np.random.uniform(0, w - self.crop_size[0] + 1)))
        crop_top = int(np.floor(np.random.uniform(0, h - self.crop_size[1] + 1)))
        hf = np.random.randint(0, 2)
        vf = np.random.randint(0, 2)
        rot = np.random.randint(0, 4)
        blurry_img = self.data_argumentation(blurry_img, crop_left, crop_top, hf, vf, rot) / 255.
        clear_img = self.data_argumentation(clear_img, crop_left, crop_top, hf, vf, rot) / 255.
        batch = {'img256': blurry_img, 'label256': clear_img}
        return batch


class TestDataset(data.Dataset):
    def __init__(self, img_list_file):
        super(type(self), self).__init__()
        st = time.time()
        f = open(img_list_file, 'r')
        lines = f.readlines()
        f.close()
        self.test_img_list = []
        self.test_gt_list = []
        for line in lines:
            img_name, gt_name = line[:-1].split(',')
            self.test_img_list.append(torch.from_numpy(np.array(cv2.imread(os.path.join(data_offset,img_name))).transpose((2, 0, 1))))
            self.test_gt_list.append(torch.from_numpy(np.array(cv2.imread(os.path.join(data_offset,gt_name))).transpose((2, 0, 1))))
        print('loading data finished', time.time() - st)

    def __len__(self):
        return len(self.test_img_list)

    def __getitem__(self, idx):
        clear_img = self.test_gt_list[idx]
        blurry_img = self.test_img_list[idx]
        # print(blurry_img.shape)
        _, h, w = blurry_img.shape
        aim_h = int(np.ceil(h / 16) * 16)
        aim_w = int(np.ceil(w / 16) * 16)
        pad_h = int((aim_h - h) / 2)
        pad_w = int((aim_w - w) / 2)
        clear_img = F.pad(clear_img, padding=[pad_w, pad_h], fill=0, padding_mode='reflect') / 255.
        blurry_img = F.pad(blurry_img, padding=[pad_w, pad_h], fill=0, padding_mode='reflect') / 255.
        # print(blurry_img.shape)
        batch = {'img256': blurry_img, 'label256': clear_img}
        return batch
