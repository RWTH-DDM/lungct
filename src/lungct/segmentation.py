
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

    return segmentation
