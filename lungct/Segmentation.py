
import nibabel as nib
import numpy as np
from scipy import ndimage, spatial


class Segmentation:

    def __init__(self, lungct, data: np.array):

        self._lungct = lungct
        self._data = data

    def get_data(self) -> np.array:

        return self._data

    def get_mask(self) -> np.array:

        mask = np.copy(self._data)
        mask[np.isnan(self._data)] = 0
        mask[mask != 0] = 1

        return mask

    def get_centroid(self, type='com') -> tuple:

        coordinates = np.argwhere(~np.isnan(self._data))

        if type == 'com':

            length = coordinates.shape[0]

            return (
                np.sum(coordinates[:, 0]) / length,
                np.sum(coordinates[:, 1]) / length,
                np.sum(coordinates[:, 2]) / length
            )

        elif type == 'median':

            return (
                np.median(coordinates[:, 0]),
                np.median(coordinates[:, 1]),
                np.median(coordinates[:, 2])
            )

        else:

            raise ValueError()

    def get_volume(self) -> float:

        return (self._data.size - np.count_nonzero(np.isnan(self._data))) * self._lungct.get_voxel_volume() / 1000000.

    def get_percentile_density(self, percentile: float) -> float:

        return np.nanpercentile(self._data, percentile)

    def get_average_density(self):

        return np.nanmean(self._data)

    def get_median_density(self):

        return np.nanmedian(self._data)

    def get_density_gradient(self) -> tuple:

        data = np.copy(self._data)

        # shift-normalize data by moving center of mass to geometric center
        centroid = (len(data) // 2, len(data[1]) // 2, len(data[0][0]) // 2)
        center_of_mass = ndimage.center_of_mass(self.get_mask())
        data = ndimage.shift(data, np.subtract(center_of_mass, centroid))

        # by replacing the data outside the segmentation with its average value we ensure that the shape of the
        # segmentation does not influence the result
        data[np.isnan(data)] = self.get_average_density()

        # divide data cuboid into octants and calculate there means
        center1, center2, center3 = len(data) // 2, len(data[0]) // 2, len(data[0][0]) // 2
        octant111 = np.mean(data[:center1, :center2, :center3])
        octant112 = np.mean(data[:center1, :center2, center3:])
        octant121 = np.mean(data[:center1, center2:, :center3])
        octant122 = np.mean(data[:center1, center2:, center3:])
        octant211 = np.mean(data[center1:, :center2, :center3])
        octant212 = np.mean(data[center1:, :center2, center3:])
        octant221 = np.mean(data[center1:, center2:, :center3])
        octant222 = np.mean(data[center1:, center2:, center3:])

        # computer gradient with finite differences
        gradient = (
            (octant211 + octant212 + octant221 + octant222) - (octant111 + octant112 + octant121 + octant122),
            (octant121 + octant122 + octant221 + octant222) - (octant111 + octant112 + octant211 + octant212),
            (octant112 + octant122 + octant212 + octant222) - (octant111 + octant121 + octant211 + octant221)
        )

        return gradient

    def get_distances_to_nearest(self, other_segmentation) -> np.array:

        """ Computes the minimal distance from each point of this segmentation to the nearest point of the given
        segmentation. Returns array of distances in mm. """

        indices = np.argwhere(~np.isnan(other_segmentation.get_data()))
        tree = spatial.KDTree(indices)

        src = np.argwhere(~np.isnan(self.get_mask()))

        # todo: parallelize code or use approximative method and remove limit
        # see: https://scipy-cookbook.readthedocs.io/items/KDTree_example.html
        # see: https://github.com/gieseke/bufferkdtree
        src = src[:10000]
        distances, _ = tree.query(src)

        # todo: map coordinate-distance to spatial distance. currently assumes mm

        return distances

    def nifti_export(self, target_file_path: str):

        if not target_file_path.endswith('.nii'):
            target_file_path += '.nii'

        nib.save(nib.Nifti1Image(self.get_data(), np.eye(4)), target_file_path)
