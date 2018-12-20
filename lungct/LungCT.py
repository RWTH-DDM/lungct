
from lungct.floodfill import flood_fill
import hashlib
import nibabel as nib
import numpy as np
import os
import scipy.ndimage as img


class LungCT:

    _cache_path = os.path.dirname(__file__) + '/../cache/'

    def __init__(self, scan_file_path: str):

        self._scan_file_path = scan_file_path
        self._scan = nib.load(scan_file_path)

        self._mask = None

    def get_scan(self) -> np.array:

        """ Returns the original scan. """

        return self._scan.get_fdata()

    def get_mask(self) -> np.array:

        """ Returns boolean mask array which is True for all voxels being part of the lung. """

        if self._mask is None:

            path_hash = hashlib.md5(self._scan_file_path.encode('utf-8')).hexdigest()
            cache_file_path = LungCT._cache_path + path_hash + '.mask.npy'

            if os.path.isfile(cache_file_path):
                self._mask = np.load(cache_file_path)
            else:
                self._mask = self._get_lung_mask(self.get_scan())
                np.save(cache_file_path, self._mask)

        return self._mask

    def get_lung(self) -> np.array:

        """ Returns the original scan data being overlayed by the lung mask. """

        masked_data = np.copy(self.get_scan())
        masked_data[~self.get_mask()] = np.nan

        return masked_data

    def get_vessel_mask(self) -> np.array:

        """ Returns boolean mask array which is True for all voxels being considered as blood vessels. """

        # Thresholding yields blood vessel point cloud within lung volume
        masked_vessel = np.copy(self.get_lung())
        masked_vessel[np.isnan(masked_vessel)] = 0
        masked_vessel[(-590 < masked_vessel) & (masked_vessel < -400)] = 1
        masked_vessel[masked_vessel != 1] = 0

        return masked_vessel.astype(bool)

    def get_lung_without_vessels(self) -> np.array:

        lung = np.copy(self.get_lung())
        lung[self.get_vessel_mask()] = np.nan

        return lung

    def get_volume(self) -> float:

        return np.count_nonzero(self.get_mask()) * self.get_voxel_volume() / 1000000.

    def get_voxel_volume(self) -> float:

        """ Returns the volume of a single voxel within the scan in mm^3. """

        header = self._scan.get_header()

        unit = header.get_xyzt_units()[0]
        unit_factors = {
            'm': 0.001,
            'mm': 1.
        }

        # Assume mm for unknown dimension
        if unit == 'unknown':
            unit = 'mm'
        elif unit not in unit_factors:
            raise Exception("Unknown unit: %s" % unit)

        # see http://nipy.org/nibabel/coordinate_systems.html#applying-the-affine
        affine = self._scan.get_affine()
        m = affine[:3, :3]
        p = m.dot([1., 1., 1.])
        p *= unit_factors[unit]

        # return volume in mm^3
        return abs(p[0] * p[1] * p[2])

    def get_percentile_density(self, percentile: float) -> float:

        return np.nanpercentile(self.get_lung(), percentile)

    def get_average_density(self):

        return np.nanmean(self.get_lung())

    def get_median_density(self):

        return np.nanmedian(self.get_lung())

    def _get_lung_mask(self, scan_data: np.array):

        # Thresholding yields point cloud within lung volume
        # Thresholds taken from https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152505
        segmentation = np.copy(scan_data)
        segmentation[(-950 < segmentation) & (segmentation < -701)] = 1
        segmentation[segmentation != 1] = 0

        # Smoothing using gaussian kernel and 2nd threshold to get solid volumes
        # todo: use otsu?
        segmentation = img.filters.gaussian_filter(segmentation, sigma=3)
        segmentation[segmentation > 0.1] = 1  # min 3 (3/27 =~ 0.11) lung-pixels per 3x3x3-cube
        segmentation[segmentation != 1] = 0

        # Use floodfilling to reduce mask to actual parts of the lung
        # The filling is started on the two points where the central traverse axis first hits a thresholded area
        shape = scan_data.shape
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

