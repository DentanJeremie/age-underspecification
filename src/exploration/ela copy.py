import imageio
import io
import scipy
import numpy as np
import cv2
from tqdm import tqdm

from src.utils.pathtools import project
from src.utils.logging import logger

QUALITY = 95
FORMAT = 'jpg'
IMAGE = 9900
MAX_PIXEL_VALUE = 255
SCALE = 0.20
SIZE = 10
THRESHOLD = 0.7
WIDTH = 110
HEIGHT = 40

path = project.human_age_unlabeled_folder / f'{IMAGE}.jpg'
output_folder = project.mkdir_if_not_exists(project.output / 'ela')
output_original = output_folder / 'original.png'
output_compressed = output_folder / 'commpressed.png'
output_difference = output_folder / 'difference.png'
output_ela_processed = output_folder / 'ela_processed.png'
output_framed = output_folder / 'framed.png'

logger.info('Reading original image')
original_image = imageio.imread(path)

logger.info('Compressing image')
compression_buffer = io.BytesIO()
imageio.imwrite(
    compression_buffer,
    original_image,
    format='jpeg',
    quality = QUALITY,
)
compressed_image = imageio.imread(compression_buffer, format=FORMAT)

# Difference image
logger.info('Computing difference')
difference_image = SCALE * cv2.absdiff(original_image, compressed_image)

# Post-processing
logger.info(f'Shape: {difference_image.shape}')
difference_image = np.sum(difference_image, axis=2)
processed_image = np.zeros((218, 178))
kernel = np.ones((SIZE * 2, SIZE * 2)) / (SIZE * 2 * SIZE * 2)


processed_image = scipy.signal.convolve2d(
    difference_image,
    kernel,
    mode = 'same',
    boundary = 'fill',
)

processed_image[processed_image<THRESHOLD*np.amax(processed_image)] = 0

# Masking
framed_image = np.copy(original_image)
idx_center, idy_center = scipy.ndimage.measurements.center_of_mass(processed_image)
idx_center, idy_center = int(idx_center), int(idy_center)

framed_image[idx_center-2:idx_center+3, idy_center-2:idy_center+3, 0] = 255
framed_image[idx_center-2:idx_center+3, idy_center-2:idy_center+3, 1] = 0
framed_image[idx_center-2:idx_center+3, idy_center-2:idy_center+3, 2] = 0

start_point = (idy_center - WIDTH//2, idx_center - HEIGHT//2)
end_point = (idy_center + WIDTH//2, idx_center + HEIGHT//2)

cv2.rectangle(framed_image, start_point, end_point, color=(255, 0, 0), thickness=3)




# Saving
imageio.imwrite(output_original, original_image, format='png')
imageio.imwrite(output_compressed, compressed_image, format='png')
imageio.imwrite(output_difference, difference_image, format='png')
imageio.imwrite(output_ela_processed, processed_image, format='png')
imageio.imwrite(output_framed, framed_image, format='png')