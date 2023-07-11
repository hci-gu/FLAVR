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

        if self.training:
            self.transforms = transforms.Compose([
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        if self.training:
            imgpath_train = os.path.join(
                self.data_root, "TOMO", self.trainlist[index])
            imgpath_gt = os.path.join(
                self.data_root, "CT", self.trainlist[index])
        else:
            imgpath_train = os.path.join(
                self.data_root, "TOMO", self.trainlist[index])
            imgpath_gt = os.path.join(
                self.data_root, "CT", self.trainlist[index])

        imgpaths_train = [imgpath_train + f'/im{i}.png' for i in range(0, 53)]
        # Only generate paths for existing ground truth images
        imgpaths_gt = [imgpath_gt + f'/im{i*10}.png' for i in range(0, 52)]

        # Load images
        images_train = [Image.open(pth) for pth in imgpaths_train]
        images_gt = [Image.open(pth) for pth in imgpaths_gt]

        # Data augmentation
        if self.training:
            seed = random.randint(0, 2**32)
            random.seed(seed)
            images_train = [self.transforms(img_) for img_ in images_train]
            random.seed(seed)
            images_gt = [self.transforms(img_) for img_ in images_gt]

            # Random Temporal Flip
            if random.random() >= 0.5:
                images_train = images_train[::-1]
                images_gt = images_gt[::-1]

        else:
            # Apply transforms for testing
            images_train = [self.transforms(img_) for img_ in images_train]
            images_gt = [self.transforms(img_) for img_ in images_gt]

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
