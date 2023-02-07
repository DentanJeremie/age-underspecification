import typing as t

import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.pathtools import project

NATURAL_IMAGE_TRANSFORM = transforms.Normalize(           
    mean=[0.485, 0.456, 0.406],      
    std=[0.229, 0.224, 0.225]   
)
DATASET_TYPES = ['hair', 'age']
HAIR_TYPE = 'hair'
AGE_TYPE = 'age'


class ImageDataset(Dataset):
    def __init__(self, type = HAIR_TYPE, labeled: bool = True):
        assert type in DATASET_TYPES, f'The dataset type must be in {DATASET_TYPES}'

        if type == HAIR_TYPE:
            if labeled:
                self.img_dir = project.human_hair_labeled_folder
                self.labels = pd.read_csv(project.human_hair_y_labeled)
            if not labeled:
                self.img_dir = project.human_hair_unlabeled_folder
                self.labels = pd.read_csv(project.human_hair_y_unlabeled)

        if type == AGE_TYPE:
            if labeled:
                self.img_dir = project.human_age_labeled_folder
                self.labels = pd.read_csv(project.human_age_y_labeled)
            if not labeled:
                self.img_dir = project.human_hair_unlabeled_folder
                self.labels = pd.read_csv(project.human_age_y_unlabeled)
                # Those labels are not relevant yet returned for consistency in the return type

        self.len = len(list(self.img_dir.iterdir()))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = self.img_dir / f"{idx}.jpg"
        image = read_image(str(img_path))
        label = self.labels.loc[idx, "labels"]
        return NATURAL_IMAGE_TRANSFORM(image.float()), label

def get_dataset(
        type = HAIR_TYPE,
        labeled: bool = True,
        val_prop = 0.5,
    ) -> t.Tuple[torch.torch.utils.data.dataset.Dataset, torch.utils.data.dataset.Dataset]:
    """Returns two datasets: one for training and one for validation.
    
    :param type: The type of dataset to get (hair or age)
    :param labeled: Whether to get the labeled dataset or not
    :param val_prop: The proportion of images to use for validation
    :returns: `train_dataset, test_dataset` Each of them are `torch.utils.data.Dataset`
    that return tuples of `tensor, int` representing an image and its label.
    """
    assert type in DATASET_TYPES, f'The dataset type must be in {DATASET_TYPES}'

    dataset = ImageDataset(type=type, labeled=labeled)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, lengths=[1-val_prop, val_prop]
    )
    return train_dataset, test_dataset

def get_dataloader(
        type = HAIR_TYPE,
        labeled: bool = True,
        val_prop = 0.5,
        batch_size = 16,
    ) -> t.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns two dataloaders: one for training and one for validation.
    
    :param type: The type of dataset to get (hair or age)
    :param labeled: Whether to get the labeled dataset or not
    :param val_prop: The proportion of images to use for validation
    :param batch_size: The size of the batches
    :returns: `train_dataloader, test_dataloader` Each of them are `torch.utils.data.DataLoader`
    that return tuples of `tensor, int` representing an image and its label.
    """
    train_dataset, test_dataset = get_dataset(type, labeled, val_prop)
    return (
        torch.utils.data.DataLoader(train_dataset, batch_size = batch_size),
        torch.utils.data.DataLoader(test_dataset, batch_size = batch_size),
    )
