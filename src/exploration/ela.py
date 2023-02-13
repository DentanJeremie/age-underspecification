import imageio
import io
import skimage
import numpy as np

from src.utils.pathtools import project
from src.utils.logging import logger

QUALITY = 90
FORMAT = 'jpg'
IMAGE = 0
MAX_PIXEL_VALUE = 255
ERROR_RANGE = 0.8

path = project.human_age_labeled_folder / f'{IMAGE}.jpg'
output_folder = project.mkdir_if_not_exists(project.output / 'ela')
output_original = output_folder / 'original.png'
output_compressed = output_folder / 'commpressed.png'
output_difference = output_folder / 'difference.png'
output_ela = output_folder / 'ela.png'
output_ela_inverted = output_folder / 'ela_inverted.png'
output_ela_processed = output_folder / 'ela_processed.png'

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
difference_image = skimage.util.compare_images(original_image, compressed_image)
difference_image = skimage.img_as_ubyte(difference_image)

# Post-processing
logger.info('Post-processing')
#get the maximum difference value
max_diff = np.amax(difference_image)

#enhance the brightness of the difference image
image_ELA = skimage.exposure.adjust_gamma(difference_image, max_diff / MAX_PIXEL_VALUE)

#calculate the inverse of the ELA image
inverted_image_ELA = skimage.util.invert(image_ELA)

# post-processing


# Saving
logger.info('Saving')
imageio.imwrite(output_original, original_image, format='png')
imageio.imwrite(output_compressed, compressed_image, format='png')
imageio.imwrite(output_difference, difference_image, format='png')
imageio.imwrite(output_ela, image_ELA, format='png')
imageio.imwrite(output_ela_inverted, inverted_image_ELA, format='png')
imageio.imwrite(output_ela_processed, clipped_ELA, format='png')