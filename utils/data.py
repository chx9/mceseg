from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import albumentations as A
import numpy as np
from torchvision.transforms import RandomResizedCrop, RandomRotation, Resize, ToTensor
from torchvision import transforms
import torchvision
import os
import cv2
from PIL import Image
ROOT_DIR = 'data'


class MceDataset(Dataset):

    def __init__(self, root_dir=ROOT_DIR, is_train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        if self.is_train:
            folder = "train"
        else:
            folder = "val"
        self.image_dir = os.path.join(root_dir, folder, 'images')
        self.label_dir = os.path.join(root_dir, folder, 'labels')
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.label_filenames = sorted(os.listdir(self.label_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        # Load image and label
        image_path = os.path.join(self.image_dir, self.image_filenames[index])
        label_path = os.path.join(self.label_dir, self.label_filenames[index])

        image = Image.open(image_path)
        label = Image.open(label_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        resize = torchvision.transforms.Resize(256)

        image = resize(image)
        label = resize(label)

        transform = A.Compose([
            A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.2)),
            A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0),]
        )
        # Apply the transformation pipeline to both the image and label
        transformed = transform(image=np.array(image), mask=np.array(label))

        # Extract the transformed image and label
        image = transformed['image']
        label = transformed['mask']
        # Convert the label to binary values (0 or 1)
        label[label > 0] = 1
        label = np.array(label, dtype=np.int64)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        image = transform(image)

        label = torch.tensor(label, dtype=torch.long)
        label = torch.squeeze(label, dim=0)
        return image, label


if __name__ == '__main__':
    mcedata = MceDataset()
    train_dataset = MceDataset(is_train=True)
    test_dataset = MceDataset(is_train=False)
    batch_size = 64
    shuffle = True
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    test_dataloder = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
