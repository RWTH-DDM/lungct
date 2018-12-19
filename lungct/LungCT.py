
import hashlib
import nibabel as nib
import numpy as np
import os
import segmentation


class LungCT:

    _cache_path = os.path.dirname(__file__) + '/../cache/'

    def __init__(self, scan_file_path: str):

        self._scan_file_path = scan_file_path
        self._scan = nib.load(scan_file_path)

        self._mask = None

    def get_scan(self) -> np.array:

        return self._scan.get_fdata()

    def get_mask(self) -> np.array:

        if self._mask is None:

            path_hash = hashlib.md5(self._scan_file_path.encode('utf-8')).hexdigest()
            cache_file_path = LungCT._cache_path + path_hash + '.mask.npy'

            if os.path.isfile(cache_file_path):
                self._mask = np.load(cache_file_path)
            else:
                self._mask = segmentation.get_lung_mask(self.get_scan())
                np.save(cache_file_path, self._mask)

        return self._mask

    def get_lung(self) -> np.array:

        masked_data = np.copy(self.get_scan())
        masked_data[~self.get_mask()] = np.nan

        return masked_data

    def get_vessel_mask(self) -> np.array:

        # Thresholding yields blood vessel point cloud within lung volume
        masked_vessel = np.copy(self.get_lung())
        masked_vessel[masked_vessel == np.nan] = 0
        masked_vessel[(-590 < masked_vessel) & (masked_vessel < -400)] = 1
        masked_vessel[masked_vessel != 1] = 0

        return masked_vessel.astype(bool)

    def get_lung_without_vessel(self) -> np.array:

        lung = np.copy(self.get_lung())
        lung[self.get_vessel_mask()] = np.nan

        return lung

    def get_volume(self) -> float:

        return np.count_nonzero(self.get_mask()) * self.get_voxel_volume() / 1000000.

    def get_voxel_volume(self) -> float:

        header = self._scan.get_header()

        unit = header.get_xyzt_units()[0]
        unit_factors = {
            'm': 0.000001,
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
