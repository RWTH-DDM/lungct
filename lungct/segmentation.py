
from lungct.floodfill import flood_fill
import numpy as np
import scipy.ndimage as img


def segment_lung(scan):

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

    first_wing = None
    for x in range(shape[2] // 2, shape[2] - 1):
        coordinates = (shape[0] // 2, shape[1] // 2, x)
        if segmentation[coordinates] == 1:
            first_wing = flood_fill(segmentation, coordinates)
            break
    if first_wing is None:
        raise Exception("Could not find first lung wing")

    second_wing = None
    for x in range(shape[2] // 2, 0, -1):
        coordinates = (shape[0] // 2, shape[1] // 2, x)
        if segmentation[coordinates] == 1:
            second_wing = flood_fill(segmentation, coordinates)
            break
    if second_wing is None:
        raise Exception("Could not find second lung wing")

    return np.logical_or(first_wing, second_wing)
