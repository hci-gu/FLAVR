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

        # self.trainlist = os.listdir(os.path.join(data_root, "TOMO"))
        self.datalist = os.listdir(os.path.join(data_root, "CT"))

        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        imgpaths = os.path.join(
            self.data_root, "CT", self.trainlist[index])
        # sort by name "imgX.png" where X is the frame number
        imgpaths = sorted(imgpaths, key=lambda x: int(x.split("img")[1].split(".")[0]))

        # take a random index from 0 to length - 8
        idx = np.random.randint(0, len(imgpaths)-8)

        # take out frame at idx and idx + 8
        imgpaths_train = [
            imgpaths[idx], 
            imgpaths[idx+3],
            imgpaths[idx+6],
            imgpaths[idx+8]
        ]

        # imgpaths_gt should contains all frames between idx and idx + 8
        imgpaths_gt = imgpaths[idx:idx+8]

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
        return len(self.datalist)


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
