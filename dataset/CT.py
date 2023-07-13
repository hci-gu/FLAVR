import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random


class CTDataset(Dataset):
    def __init__(self, data_root, is_training):
        """
        Creates a Vimeo Septuplet object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
            input_frames: Which frames to input for frame interpolation network.
        """
        self.data_root = data_root
        self.training = is_training

        self.trainlist = os.listdir(os.path.join(data_root, "TOMO"))
        self.testlist = os.listdir(os.path.join(data_root, "CT"))

        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        imgpath_train = os.path.join(
            self.data_root, "TOMO", self.trainlist[index])
        imgpath_gt = os.path.join(
            self.data_root, "CT", self.trainlist[index])

        imgpaths_train = [imgpath_train + f'/im{i}.png' for i in range(0, 63)]
        # Only generate paths for existing ground truth images
        imgpaths_gt = [imgpath_gt + f'/im{i}.png' for i in range(0, 512)]

        # take two random frames in sequence from the training set
        idx = np.random.randint(0, len(imgpaths_train)-1)
        imgpaths_train = imgpaths_train[idx:idx+2]

        # take the corresponding ground truth frames which are the 10 frames after the first frame
        gt_idx = idx*8
        imgpaths_gt = imgpaths_gt[gt_idx:gt_idx+8]

        # take third and sixth frame from ground truth frames and insert to training frames
        # [training frame 1, gt at index 3, gt at index 6, training frame 2]
        imgpaths_train.insert(1, imgpaths_gt[3])
        imgpaths_train.insert(2, imgpaths_gt[5])

        # Load images
        images_train = [Image.open(pth) for pth in imgpaths_train]
        images_gt = [Image.open(pth) for pth in imgpaths_gt]

        # Data augmentation
        images_train = [self.transforms(img_) for img_ in images_train]
        images_gt = [self.transforms(img_) for img_ in images_gt]
        if self.training:
            if random.random() >= 0.5:
                images_train = images_train[::-1]
                images_gt = images_gt[::-1]

        return images_train, images_gt

    def __len__(self):
        if self.training:
            return len(self.trainlist)
        else:
            return len(self.testlist)


def get_loader(mode, data_root, batch_size, shuffle, num_workers, test_mode=None):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = CTDataset(data_root, is_training=is_training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":

    dataset = CTDataset("./data/compiled", is_training=True)
    dataloader = DataLoader(dataset, batch_size=100,
                            shuffle=False, num_workers=32, pin_memory=True)
