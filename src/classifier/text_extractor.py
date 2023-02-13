import os
import pickle
import io

import cv2
import imageio
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from p_tqdm import p_map
import scipy
import torch
from tqdm import tqdm

from src.utils.pathtools import project
from src.utils.logging import logger

LABELED_FILE_PATH = project.text_frames_dir / 'labeled.obj'
UNLABELED_FILE_PATH = project.text_frames_dir / 'unlabeled.obj'

COMPRESSION_QUALITY = 95
FORMAT = 'jpg'
SMOOTHING_SIZE = 20
THRESHOLD = 0.7
MASK_WIDTH = 110
MASK_HEIGHT = 40
ORIGINAL_IMAGES_HEIGHT = 218
ORIGINAL_IMAGES_WIDTH = 178

# ------------------ DATASET CLASS FOR ELA ------------------

class ImageDatasetForELA():
    def __init__(
            self,
            labeled: bool = True,
        ):

        if labeled:
            self.img_dir = project.human_age_labeled_folder
            self.labels = pd.read_csv(project.human_age_y_labeled)
        if not labeled:
            self.img_dir = project.human_age_unlabeled_folder
            self.labels = pd.read_csv(project.human_age_y_unlabeled)
                # Those labels are not relevant yet returned for consistency in the return type

        self.len = len(list(self.img_dir.iterdir()))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = self.img_dir / f"{idx}.jpg"
        image = imageio.imread(img_path)

        return image

# ------------------ PROCESSING OF A SINGLE IMAGE ------------------

def get_frame_center(img_original):
    # Compression
    compression_buffer = io.BytesIO()
    imageio.imwrite(
        compression_buffer,
        np.asarray(img_original, np.uint8),
        format='jpeg',
        quality = COMPRESSION_QUALITY,
    )
    img_compressed = imageio.imread(compression_buffer, format=FORMAT)

    # Difference image
    img_original = np.asarray(img_original, np.float32)
    img_compressed = np.asarray(img_compressed, np.float32)
    img_diff = cv2.absdiff(img_original, img_compressed)
    img_diff = np.sum(img_diff, axis=2)

    # Soothing
    kernel = np.ones((SMOOTHING_SIZE, SMOOTHING_SIZE)) / (SMOOTHING_SIZE * SMOOTHING_SIZE)
    img_processed = scipy.signal.convolve2d(
        img_diff,
        kernel,
        mode = 'same',
        boundary = 'fill',
    )

    # Clipping, center of mass
    img_processed[img_processed<THRESHOLD*np.amax(img_processed)] = 0
    idx_center, idy_center = scipy.ndimage.measurements.center_of_mass(img_processed)
    idx_center, idy_center = int(idx_center), int(idy_center)
    return idx_center, idy_center

# ------------------ PROCESSING OF A SINGLE IMAGE ------------------

class TextExtractor():
    def __init__(self, labeled:bool):
        logger.info(f'Creating a text extractor, labeled={labeled}')
        self.dataset = ImageDatasetForELA(labeled=labeled)
        self._frame_centers = dict()
        self.labeled = labeled
        self.dict_path = LABELED_FILE_PATH if labeled else UNLABELED_FILE_PATH

        logger.info('Checking on disk if some text centers have already been computed')
        if self.dict_path.exists():
            logger.info('Found a file for labled images')
            try:
                with self.dict_path.open('rb') as f:
                    self._frame_centers = pickle.load(f)
                if type(self._frame_centers) != dict:
                    raise ValueError
                logger.info('Successfully read the dict from disk.')
            except:
                logger.warn(f'Unable to read from file {project.as_relative(self.dict_path)}')
                self._frame_centers = dict()

    @property
    def frame_centers(self):
        if len(self._frame_centers) == 0:
            self.build_frames()
        return self._frame_centers

    def save_frames(self):
        with self.dict_path.open('wb') as f:
            pickle.dump(self._frame_centers, f)

    def build_frames(self):
        """
        Builds the frames of the given dataset.
        """
        logger.info('Reading data from disk')
        dataset_list = list()
        for index in tqdm(range(len(self.dataset))):
            img_original = self.dataset[index]
            dataset_list.append(img_original)

        logger.info(f'Starting computing text frames using {os.cpu_count()} cores')
        text_centers = p_map(get_frame_center, dataset_list)

        logger.info('Collecting results and converting into a dict')
        for index, (idx_center, idy_center) in enumerate(text_centers):
            self._frame_centers[index] = (idx_center, idy_center)

        self.save_frames()

    def get_mask(self, image_index:int) -> torch.tensor:
        """
        Returns a mask of the same size as the image, with 0 where the text has been detected.
        """
        mask = torch.ones(3, ORIGINAL_IMAGES_HEIGHT, ORIGINAL_IMAGES_WIDTH)
        idx_center, idy_center = self.frame_centers[image_index]

        mask[
            :,
            max(idx_center - MASK_HEIGHT//2, 0):min(idx_center + MASK_HEIGHT//2, ORIGINAL_IMAGES_HEIGHT),
            max(idy_center - MASK_WIDTH//2, 0):min(idy_center + MASK_WIDTH//2, ORIGINAL_IMAGES_WIDTH),
        ] = 0

        return mask 

    def plot_frame(self, image_index=0):

        # Loading image
        img_original = self.dataset[image_index]

        # Framing
        idx_center, idy_center = self.frame_centers[image_index]

        img_original[idx_center-2:idx_center+3, idy_center-2:idy_center+3, 0] = 255
        img_original[idx_center-2:idx_center+3, idy_center-2:idy_center+3, 1] = 0
        img_original[idx_center-2:idx_center+3, idy_center-2:idy_center+3, 2] = 0

        start_point = (idy_center - MASK_WIDTH//2, idx_center - MASK_HEIGHT//2)
        end_point = (idy_center + MASK_WIDTH//2, idx_center + MASK_HEIGHT//2)

        cv2.rectangle(img_original, start_point, end_point, color=(255, 0, 0), thickness=3)

        plt.imshow(img_original)
        plt.show()


if __name__ == '__main__':
    te = TextExtractor(labeled = False)
    te.plot_frame(image_index=9)