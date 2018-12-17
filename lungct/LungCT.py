
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

    def get_volume(self) -> float:

        return np.count_nonzero(self.get_mask()) * self.get_voxel_volume()

    def get_voxel_volume(self) -> float:

        header = self._scan.get_header()

        unit = header.get_xyzt_units()[0]
        unit_factors = {
            'm': 0.000001,
            'mm': 1.
        }

        if unit not in unit_factors:
            raise Exception("Unknown unit: %s" % unit[0])

        # see http://nipy.org/nibabel/coordinate_systems.html#applying-the-affine
        affine = self._scan.get_affine()
        m = affine[:3, :3]
        p = m.dot([1., 1., 1.])
        p *= unit_factors[unit]

        return abs(p[0] * p[1] * p[2])
