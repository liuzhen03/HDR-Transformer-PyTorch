#-*- coding:utf-8 -*-
import os
import os.path as osp
import sys
sys.path.append('..')
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.utils import *


class SIG17_Training_Dataset(Dataset):

    def __init__(self, root_dir, sub_set, is_training=True):
        self.root_dir = root_dir
        self.is_training = is_training
        self.sub_set = sub_set

        self.scenes_dir = osp.join(root_dir, self.sub_set)
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            exposure_file_path = os.path.join(self.scenes_dir, self.scenes_list[scene], 'exposure.txt')
            ldr_file_path = list_all_files_sorted(os.path.join(self.scenes_dir, self.scenes_list[scene]), '.tif')
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path]]

    def __getitem__(self, index):
        # Read exposure times
        expoTimes = read_expo_times(self.image_list[index][0])
        # Read LDR images
        ldr_images = read_images(self.image_list[index][1])
        # Read HDR label
        label = read_label(self.image_list[index][2], 'label.hdr') # 'label.hdr' for cropped training data
        # ldr images process
        pre_img0 = ldr_to_hdr(ldr_images[0], expoTimes[0], 2.2)
        pre_img1 = ldr_to_hdr(ldr_images[1], expoTimes[1], 2.2)
        pre_img2 = ldr_to_hdr(ldr_images[2], expoTimes[2], 2.2)

        pre_img0 = np.concatenate((pre_img0, ldr_images[0]), 2)
        pre_img1 = np.concatenate((pre_img1, ldr_images[1]), 2)
        pre_img2 = np.concatenate((pre_img2, ldr_images[2]), 2)

        img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
        img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
        img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
        label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)

        sample = {
            'input0': img0, 
            'input1': img1, 
            'input2': img2, 
            'label': label
            }
        return sample

    def __len__(self):
        return len(self.scenes_list)


class SIG17_Validation_Dataset(Dataset):

    def __init__(self, root_dir, is_training=False, crop=True, crop_size=512):
        self.root_dir = root_dir
        self.is_training = is_training
        self.crop = crop
        self.crop_size = crop_size

        # sample dir
        self.scenes_dir = osp.join(root_dir, 'Test')
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            exposure_file_path = os.path.join(self.scenes_dir, self.scenes_list[scene], 'exposure.txt')
            ldr_file_path = list_all_files_sorted(os.path.join(self.scenes_dir, self.scenes_list[scene]), '.tif')
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path]]

    def __getitem__(self, index):
        # Read exposure times
        expoTimes = read_expo_times(self.image_list[index][0])
        # Read LDR images
        ldr_images = read_images(self.image_list[index][1])
        # Read HDR label
        label = read_label(self.image_list[index][2], 'HDRImg.hdr') # 'HDRImg.hdr' for test data
        # ldr images process
        pre_img0 = ldr_to_hdr(ldr_images[0], expoTimes[0], 2.2)
        pre_img1 = ldr_to_hdr(ldr_images[1], expoTimes[1], 2.2)
        pre_img2 = ldr_to_hdr(ldr_images[2], expoTimes[2], 2.2)

        # concat: linear domain + ldr domain
        pre_img0 = np.concatenate((pre_img0, ldr_images[0]), 2)
        pre_img1 = np.concatenate((pre_img1, ldr_images[1]), 2)
        pre_img2 = np.concatenate((pre_img2, ldr_images[2]), 2)

        if self.crop:
            x = 0
            y = 0
            img0 = pre_img0[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
            img1 = pre_img1[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
            img2 = pre_img2[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
            label = label[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
        else:
            img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
            img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
            img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
            label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)

        sample = {
            'input0': img0, 
            'input1': img1, 
            'input2': img2, 
            'label': label
            }
        return sample

    def __len__(self):
        return len(self.scenes_list)


class Img_Dataset(Dataset):
    def __init__(self, ldr_path, label_path, exposure_path, patch_size):
        self.ldr_images = read_images(ldr_path)
        self.label = read_label(label_path, 'HDRImg.hdr')
        self.ldr_patches = self.get_ordered_patches(patch_size)
        self.expo_times = read_expo_times(exposure_path)
        self.patch_size = patch_size
        self.result = []

    def __getitem__(self, index):
        pre_img0 = ldr_to_hdr(self.ldr_patches[index][0], self.expo_times[0], 2.2)
        pre_img1 = ldr_to_hdr(self.ldr_patches[index][1], self.expo_times[1], 2.2)
        pre_img2 = ldr_to_hdr(self.ldr_patches[index][2], self.expo_times[2], 2.2)
        pre_img0 = np.concatenate((pre_img0, self.ldr_patches[index][0]), 2)
        pre_img1 = np.concatenate((pre_img1, self.ldr_patches[index][1]), 2)
        pre_img2 = np.concatenate((pre_img2, self.ldr_patches[index][2]), 2)
        img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
        img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
        img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        sample = {
            'input0': img0, 
            'input1': img1, 
            'input2': img2}
        return sample

    def get_ordered_patches(self, patch_size):
        ldr_patch_list = []
        h, w, c = self.label.shape
        n_h = h // patch_size + 1
        n_w = w // patch_size + 1
        tmp_h = n_h * patch_size
        tmp_w = n_w * patch_size
        tmp_label = np.ones((tmp_h, tmp_w, c), dtype=np.float32)
        tmp_ldr0 = np.ones((tmp_h, tmp_w, c), dtype=np.float32)
        tmp_ldr1 = np.ones((tmp_h, tmp_w, c), dtype=np.float32)
        tmp_ldr2 = np.ones((tmp_h, tmp_w, c), dtype=np.float32)
        tmp_label[:h, :w] = self.label
        tmp_ldr0[:h, :w] = self.ldr_images[0]
        tmp_ldr1[:h, :w] = self.ldr_images[1]
        tmp_ldr2[:h, :w] = self.ldr_images[2]

        for x in range(n_w):
            for y in range(n_h):
                if (x+1) * patch_size <= tmp_w and (y+1) * patch_size <= tmp_h:
                    temp_patch_ldr0 = tmp_ldr0[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size]
                    temp_patch_ldr1 = tmp_ldr1[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size]
                    temp_patch_ldr2 = tmp_ldr2[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size]
                    ldr_patch_list.append([temp_patch_ldr0, temp_patch_ldr1, temp_patch_ldr2])

        assert len(ldr_patch_list) == n_h * n_w
        return ldr_patch_list

    def __len__(self):
        return len(self.ldr_patches)

    def rebuild_result(self):
        h, w, c = self.label.shape
        n_h = h // self.patch_size + 1
        n_w = w // self.patch_size + 1
        tmp_h = n_h * self.patch_size
        tmp_w = n_w * self.patch_size
        pred = np.empty((c, tmp_h, tmp_w), dtype=np.float32)

        for x in range(n_w):
            for y in range(n_h):
                pred[:, y*self.patch_size:(y+1)*self.patch_size, x*self.patch_size:(x+1)*self.patch_size] = self.result[x*n_h+y]
        return pred[:, :h, :w], self.label.transpose(2, 0, 1)

    def update_result(self, tensor):
        self.result.append(tensor)


def SIG17_Test_Dataset(root_dir, patch_size):
    scenes_dir = osp.join(root_dir, 'Test')
    scenes_list = sorted(os.listdir(scenes_dir))
    ldr_list = []
    label_list = []
    expo_times_list = []
    for scene in range(len(scenes_list)):
        exposure_file_path = os.path.join(scenes_dir, scenes_list[scene], 'exposure.txt')
        ldr_file_path = list_all_files_sorted(os.path.join(scenes_dir, scenes_list[scene]), '.tif')
        label_path = os.path.join(scenes_dir, scenes_list[scene])
        expo_times_list += [exposure_file_path]
        ldr_list += [ldr_file_path]
        label_list += [label_path]
    for ldr_dir, label_dir, expo_times_dir in zip(ldr_list, label_list, expo_times_list):
        yield Img_Dataset(ldr_dir, label_dir, expo_times_dir, patch_size)



