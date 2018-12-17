
import nibabel as nib
import numpy as np
import segmentation


class LungCT:

    def __init__(self, scan_file_path: str):

        self._scan = nib.load(scan_file_path)

        self._mask = None

    def get_scan(self) -> np.array:

        return self._scan.get_fdata()

    def get_mask(self) -> np.array:

        if self._mask is None:
            self._mask = segmentation.get_lung_mask(self.get_scan())

        return self._mask

    def get_volume(self) -> float:

        return np.count_nonzero(self.get_mask())
