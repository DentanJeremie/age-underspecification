import typing as t

import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.pathtools import project
from src.utils.logging import logger
from src.classifier.text_extractor import TextExtractor

NATURAL_IMAGE_TRANSFORM = transforms.Normalize(           
    mean=[0.5045, 0.4218, 0.3822],      
    std=[0.2680, 0.2473, 0.2425],   
)

# Consitency in the train/test over models
ENCODING_RANGE = 255
VAL_PROP = 0.2
SPLIT_SEED = 42

# ------------------ AUGMENTATION TRANSFORMS ------------------

class AddNoise(object):
    def __init__(self, noise_size:float):
        self.noise_size = noise_size
        
    def __call__(self, tensor):
        tensor += torch.randn(tensor.size()) * self.noise_size - self.noise_size/2
        tensor = torch.clamp(tensor, 0, 1)
        return tensor

class MaskText(object):
    def __init__(self, mask):
        self.mask = mask
        
    def __call__(self, tensor):
        return tensor*self.mask

# ------------------ DATASET CLASS ------------------

class ImageDataset(Dataset):
    def __init__(
            self,
            labeled: bool = True,
            noise_size:float = 0,
            mask_text:bool = False,
        ):
        """
        :param noise_size: The size of the noise to be added to each images.
        :param mask_text: Whether of not to mask the text.
        """

        self.labeled = labeled
        if self.labeled:
            self.img_dir = project.human_age_labeled_folder
            self.labels = pd.read_csv(project.human_age_y_labeled)
        if not self.labeled:
            self.img_dir = project.human_age_unlabeled_folder
            self.labels = pd.read_csv(project.human_age_y_unlabeled)
                # Those labels are not relevant yet returned for consistency in the return type

        self.len = len(list(self.img_dir.iterdir()))
        self.noise_size = noise_size

        self.mask_text = mask_text
        if self.mask_text:
            self.text_extractor = TextExtractor(labeled=self.labeled)

        logger.info(f"Builded dataset: len={self.len}, labeled={labeled}, noise_size={noise_size}, mask_text={mask_text}")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = self.img_dir / f"{idx}.jpg"
        image = read_image(str(img_path))
        label = self.labels.loc[idx, "labels"]
        transforms_to_apply = [NATURAL_IMAGE_TRANSFORM]
        if self.noise_size != 0.0:
            transforms_to_apply.append(
                AddNoise(self.noise_size)
            )
        if self.mask_text:
            transforms_to_apply.append(
                MaskText(self.text_extractor.get_mask(image_index=idx))
            )
        final_transformation = transforms.Compose(transforms_to_apply)
        return final_transformation(image.float()/ENCODING_RANGE), label


# ------------------ DATASET/DATALOADER BUILD ------------------


def get_dataset(
        labeled: bool = True,
        noise_size: float = 0.0,
        mask_text:bool = False,
    ) -> t.Tuple[torch.torch.utils.data.dataset.Dataset, torch.utils.data.dataset.Dataset]:
    """Returns two datasets: one for training and one for validation.
    
    :param labeled: Whether to get the labeled dataset or not
    :param noise_size: The size of the noise to be added to each images.
    :param mask_text: Whether of not to mask the text.
    :returns: `train_dataset, test_dataset` Each of them are `torch.utils.data.Dataset`
    that return tuples of `tensor, int` representing an image and its label.
    """
    dataset = ImageDataset(
        labeled=labeled,
        noise_size=noise_size,
        mask_text=mask_text,
    )
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, lengths=[1-VAL_PROP, VAL_PROP],
        generator = torch.Generator().manual_seed(SPLIT_SEED),
    )
    logger.info(f'Splitted dataset in with test_prop={VAL_PROP}')
    return train_dataset, test_dataset
