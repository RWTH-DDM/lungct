
import nibabel as nib
import numpy as np


class Segmentation:

    def __init__(self, lungct, data: np.array):

        self._lungct = lungct
        self._data = data

    def get_data(self) -> np.array:

        return self._data

    def get_volume(self) -> float:

        return (self._data.size - np.count_nonzero(np.isnan(self._data))) * self._lungct.get_voxel_volume() / 1000000.

    def get_percentile_density(self, percentile: float) -> float:

        return np.nanpercentile(self._data, percentile)

    def get_average_density(self):

        return np.nanmean(self._data)

    def get_median_density(self):

        return np.nanmedian(self._data)

    def nifti_export(self, target_file_path: str):

        if not target_file_path.endswith('.nii'):
            target_file_path += '.nii'

        nib.save(nib.Nifti1Image(self.get_data(), np.eye(4)), target_file_path)
