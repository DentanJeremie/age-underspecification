import imageio
import io
import scipy
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

from src.utils.pathtools import project
from src.utils.logging import logger




class TextExtractor():
    def __init__(self):
        # Output directories
        self.age_labeled_frames = project.mkdir_if_not_exists(project.text_frames_dir / 'age_labeled')
        self.age_unlabeled_frames = project.mkdir_if_not_exists(project.text_frames_dir / 'age_unlabeled')

    def build_frames(in_dir:Path, out_dir:Path):
        """
        Builds the frames for images in `in_dir` and stored them in `out_dir`.
        """
        logger.info(f'Extracting text frames {project.as_relative(in_dir)} -> {project.as_relative(in_dir)}')
