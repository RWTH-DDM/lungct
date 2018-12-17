
from lungct.floodfill import flood_fill
import numpy as np
import scipy.ndimage as img


def get_lung_mask(scan):

    # Thresholding yields point cloud within lung volume
    # Thresholds taken from https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152505
    segmentation = np.copy(scan.get_fdata())
    segmentation[(-950 < segmentation) & (segmentation < -701)] = 1
    segmentation[segmentation != 1] = 0

    # Smoothing using gaussian kernel and 2nd threshold to get solid volumes
    # todo: use otsu?
    segmentation = img.filters.gaussian_filter(segmentation, sigma=3)
    segmentation[segmentation > 0.1] = 1  # min 3 (3/27 =~ 0.11) lung-pixels per 3x3x3-cube
    segmentation[segmentation != 1] = 0

    # Use floodfilling to reduce mask to actual parts of the lung
    # The filling is started on the two points where the central traverse axis first hits a thresholded area
    shape = scan.shape
    result = np.zeros(shape, bool)

    limit = shape[2] - 1
    for x in range(shape[2] // 2, limit):

        coordinates = (shape[0] // 2, shape[1] // 2, x)

        if segmentation[coordinates] == 1:
            flood_fill(segmentation, coordinates, result)
            break

        elif x == (limit - 1):
            raise Exception("Could not find first lung wing")

    limit = 0
    for x in range(shape[2] // 2, limit, -1):

        coordinates = (shape[0] // 2, shape[1] // 2, x)

        if segmentation[coordinates] == 1:

            # only fill if not already found by first flooding
            if not result[coordinates]:
                flood_fill(segmentation, coordinates, result)

            break

        elif x == (limit + 1):
            raise Exception("Could not find second lung wing")

    return result
