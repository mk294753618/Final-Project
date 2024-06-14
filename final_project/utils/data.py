from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as func
import torch
from torchvision.transforms import transforms
import numpy as np

import os
import glob
import random
import warnings

from PIL import Image, ImageFile
from .utils import worker_seed_set
from .transforms import AUGMENTATION_TRANSFORMS, DEFAULT_TRANSFORMS, Resize


def resize(image, size):
    image = func.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ListDataset(Dataset):
    def __init__(self, path, img_size=416, multiscale=True, transform=None):
        with open(path, 'r') as f:
            self.img_file = f.readlines()

        self.label_file = []
        for p in self.img_file:
            img_dir = os.path.dirname(p)
            img_dir_basename = os.path.basename(img_dir)
            if img_dir_basename == 'train_image':
                label_dir = 'train_label'.join(img_dir.split(img_dir_basename, 1))
            elif img_dir_basename == 'valid_image':
                label_dir = 'valid_label'.join(img_dir.split(img_dir_basename, 1))
            assert label_dir != img_dir, \
                f"Image path must contain a folder named 'trainself.label_files.append(label_file)_image' or 'valid_image'! \n'{img_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(p))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_file.append(label_file)

        self.img_size = img_size
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __len__(self):
        return len(self.img_file)

    def __getitem__(self, idx):
        try:
            img_path = self.img_file[idx].rstrip()
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f'could not read image {img_path}')
            return

        try:
            label_path = self.label_file[idx].rstrip()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                box = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            print(f'could not read image {label_path}')
            return

        if self.transform:
            try:
                img, target = self.transform((img, box))
            except Exception:
                print("Could not apply transform.")
                return

        return img_path, img, target

    def collate_fn(self, batch):
        self.batch_count += 1
        batch = [data for data in batch if data is not None]
        path, imgs, boxes = list(zip(*batch))

        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size+1, 32))

        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        for i, box in enumerate(boxes):
            box[:, 0] = i

        box = torch.cat(boxes, 0)

        return path, imgs, box


def create_data_loader(path, batch_size, img_size, n_cpu, multiscale_training=False):
    dataset = ListDataset(
        path=path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set
    )
    return dataloader


def create_valid_data_loader(path, batch_size, img_size, n_cpu):
    dataset = ListDataset(
        path=path,
        img_size=img_size,
        multiscale=False,
        transform=DEFAULT_TRANSFORMS
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    return dataloader


class ImageFolder(Dataset):

    def __init__(self, path, transform=None):
        self.file = sorted(glob.glob('%s/*.*' % path))
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.file[idx]
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        box = np.zeros((1, 5))

        if self.transform:
            img, _ = self.transform((img, box))

        return img_path, img

    def __len__(self):
        return len(self.file)


def create_data_loader_detect(img_path, batch_size, img_size, n_cpu):

    dataset = ImageFolder(
        img_path,
        transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True)
    return dataloader






