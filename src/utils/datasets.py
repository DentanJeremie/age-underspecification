import typing as t

import numpy as np
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.pathtools import project
from src.utils.logging import logger

NATURAL_IMAGE_TRANSFORM = transforms.Normalize(           
    mean=[0.485, 0.456, 0.406],      
    std=[0.229, 0.224, 0.225]   
)
DATASET_TYPES = ['hair', 'age']
HAIR_TYPE = 'hair'
AGE_TYPE = 'age'
AGE_HAIR_SIZE_RATIO = 10
DEFAULT_DATA_AUGMENTATION_AGE = 2
DEFAULT_DATA_AUGMENTATION_HAIR = DEFAULT_DATA_AUGMENTATION_AGE * AGE_HAIR_SIZE_RATIO
DEFAULT_NOISE_SIZE = 0.1
DEFAULT_BATCH_SIZE = 32
LABEL_OFFSET_HAIR_TEXT = 2

# Consitency in the train/test over models
VAL_PROP = 0.2
SPLIT_SEED = 42

# ------------------ AUGMENTATION TRANSFORMS ------------------

class AddNoise(object):
    def __init__(self, noise_size=DEFAULT_NOISE_SIZE):
        self.noise_size = noise_size
        
    def __call__(self, tensor):
        tensor += torch.randn(tensor.size()) * self.noise_size - self.noise_size/2
        tensor = torch.clamp(tensor, 0, 1)
        return tensor

class RollColors(object):
    def __call__(self, tensor):
        roll_number = np.random.randint(0,3)
        return torch.roll(tensor, shifts=roll_number, dims=1)


class InvertColors(object):
    def __call__(self, tensor):
        if np.random.randint(2) == 1:
            return 1 - tensor
        return tensor


# ------------------ DATASET CLASS ------------------


class ImageDataset(Dataset):
    def __init__(
            self,
            type = HAIR_TYPE,
            labeled: bool = True,
            label_offset: int = 0,
            noise_size:float = 0,
            roll_colors: bool = False,
            invert_colors: bool = False,
        ):
        """
        :param label_offset: An offset that will be added to all labels.
        :param noise_size: The size of the noise to be added to each images.
        :param roll_colors: A boolean indicating whether or not to roll the colors.
        :param invert_colors: A boolean indicating whether or not to invert the colors.
        """
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
        self.label_offset = label_offset
        self.noise_size = noise_size
        self.roll_colors = roll_colors
        self.invert_colors = invert_colors

        logger.info(f"Builded dataset: len={self.len}, type={type}, labeled={labeled}, label_offset={label_offset}, noise_size={noise_size}, roll_colors={roll_colors}, inver_colors={invert_colors}")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = self.img_dir / f"{idx}.jpg"
        image = read_image(str(img_path))
        label = self.labels.loc[idx, "labels"] + self.label_offset
        transforms_to_apply = [NATURAL_IMAGE_TRANSFORM]
        if self.noise_size != 0.0:
            transforms_to_apply.append(
                AddNoise(self.noise_size)
            )
        if self.roll_colors:
            transforms_to_apply.append(
                RollColors()
            )
        if self.invert_colors:
            transforms_to_apply.append(
                InvertColors()
            )
        final_transformation = transforms.Compose(transforms_to_apply)
        return final_transformation(image.float()), label


# ------------------ DATASET/DATALOADER BUILD ------------------


def get_dataset(
        type: str = HAIR_TYPE,
        labeled: bool = True,
        label_offset: int = 0,
        noise_size: float = 0.0,
        roll_colors: bool = False,
        invert_colors: bool = False,
    ) -> t.Tuple[torch.torch.utils.data.dataset.Dataset, torch.utils.data.dataset.Dataset]:
    """Returns two datasets: one for training and one for validation.
    
    :param type: The type of dataset to get (hair or age)
    :param labeled: Whether to get the labeled dataset or not
    :param label_offset: The offset to be added to the labels
    :param noise_size: The size of the noise to be added to each images.
    :param roll_colors: A boolean indicating whether or not to roll the colors.
    :param invert_colors: A boolean indicating whether or not to invert the colors.
    :returns: `train_dataset, test_dataset` Each of them are `torch.utils.data.Dataset`
    that return tuples of `tensor, int` representing an image and its label.
    """
    assert type in DATASET_TYPES, f'The dataset type must be in {DATASET_TYPES}'

    dataset = ImageDataset(
        type=type,
        labeled=labeled,
        label_offset=label_offset,
        noise_size=noise_size,
        roll_colors=roll_colors,
        invert_colors=invert_colors
    )
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, lengths=[1-VAL_PROP, VAL_PROP],
        generator = torch.Generator().manual_seed(SPLIT_SEED),
    )
    logger.info(f'Splitted dataset in with test_prop={VAL_PROP}')
    return train_dataset, test_dataset

def get_dataloader(
        type: str = HAIR_TYPE,
        labeled: bool = True,
        label_offset: int = 0,
        noise_size: float = 0.0,
        roll_colors: bool = False,
        invert_colors: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> t.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns two dataloaders: one for training and one for validation.
    
    :param type: The type of dataset to get (hair or age)
    :param labeled: Whether to get the labeled dataset or not
    :param label_offset: The offset to be added to the labels
    :param noise_size: The size of the noise to be added to each images.
    :param roll_colors: A boolean indicating whether or not to roll the colors.
    :param invert_colors: A boolean indicating whether or not to invert the colors.
    :param batch_size: The size of the batches
    :returns: `train_dataloader, test_dataloader` Each of them are `torch.utils.data.DataLoader`
    that return tuples of `tensor, int` representing an image and its label.
    """
    train_dataset, test_dataset = get_dataset(
        type=type,
        labeled=labeled,
        label_offset=label_offset,
        noise_size=noise_size,
        roll_colors=roll_colors,
        invert_colors=invert_colors,
    )
    logger.info(f'Build dataloader in with batch_size={batch_size}')
    return (
        torch.utils.data.DataLoader(train_dataset, batch_size = batch_size),
        torch.utils.data.DataLoader(test_dataset, batch_size = batch_size),
    )


# ------------------ TEXT HEAD DATALOADER ------------------


def get_dataloader_text(
    augmentation_factor_hair: int = DEFAULT_DATA_AUGMENTATION_HAIR,
    augmentation_factor_age: int = DEFAULT_DATA_AUGMENTATION_AGE,
    noise_size: float = DEFAULT_NOISE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> t.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns `train_loader_text` and `test_loader_text`, two dataloaders that contain images 
    from labeled datasets of human_age and labeled datasets of human_hairs.

    :param augmentation_factor_hair: We'll add `(augmentation_factor_hair - 1) * len(human_hair_labeled)``
    noisy images of human_hair
    :param augmentation_factor_age: We'll add `(augmentation_factor_age - 1) * len(human_age_labeled)``
    noisy images of human_age
    :param batch_size: The size of the batches for the dataloader
    :param noise_size: The max size of the noise to add
    """
    list_train_dataset = list()
    list_test_dataset = list()

    # For the test set, we never use augmented data, but only real data
    # This is why we only use the test set for the first iteration in the loop, where no augmentation is done
    # During the other iterations, we use augmentation transformations: rolling colors, color inversion, noise, etc

    # Data augmentation for hairs
    # We add an offset in the labels of the hairs of `LABEL_OFFSET_HAIR_TEXT`
    logger.info(f'Starting data augmentatino of hairs of factor {augmentation_factor_hair}')
    for noise, roll, invert, use_for_test in zip(
        [0.0] + (augmentation_factor_hair - 1)*[noise_size],
        [False] + (augmentation_factor_hair - 1)*[True],
        [False] + (augmentation_factor_hair - 1)*[True],
        [True] + (augmentation_factor_hair - 1)*[False],
    ):
        train, test = get_dataset(
            type=HAIR_TYPE,
            labeled=True,
            label_offset=LABEL_OFFSET_HAIR_TEXT,
            noise_size=noise,
            roll_colors=roll,
            invert_colors=invert,
        )
        list_train_dataset.append(train)

        if use_for_test:
            for _ in range(AGE_HAIR_SIZE_RATIO):
                list_test_dataset.append(test)

    # Data augmentation for ages
    logger.info(f'Starting data augmentation of ages of factor {augmentation_factor_age}')
    for noise, roll, invert, use_for_test in zip(
        [0.0] + (augmentation_factor_age - 1)*[noise_size],
        [False] + (augmentation_factor_age - 1)*[True],
        [False] + (augmentation_factor_age - 1)*[True],
        [True] + (augmentation_factor_age - 1)*[False],
    ):
        train, test = get_dataset(
            type=AGE_TYPE,
            labeled=True,
            label_offset=0,
            noise_size=noise,
            roll_colors=roll,
            invert_colors=invert,
        )
        list_train_dataset.append(train)

        if use_for_test:
            list_test_dataset.append(test)

    # Concatenating
    logger.info(f'Concatenating {len(list_train_dataset)} train datasets and {len(list_test_dataset)} test datasets')
    full_train_dataset = torch.utils.data.ConcatDataset(list_train_dataset)
    full_test_dataset = torch.utils.data.ConcatDataset(list_test_dataset)

    logger.info(f'Building dataloaders with batch_size={batch_size}')
    full_train_dataloader = torch.utils.data.DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
    full_test_dataloader = torch.utils.data.DataLoader(full_test_dataset, batch_size=batch_size, shuffle=True)

    return full_train_dataloader, full_test_dataloader
