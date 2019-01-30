
from lungct.floodfill import flood_fill
from lungct.NumpyCache import NumpyCache
from lungct.Segmentation import Segmentation
import nibabel as nib
import numpy as np
import os
import scipy.ndimage as img
from sklearn.cluster import KMeans
from typing import Tuple


class LungCT:

    def __init__(self, scan_file_path: str):

        self._scan_file_path = scan_file_path
        self._scan = nib.load(scan_file_path)

        self._numpy_cache = NumpyCache(
            os.path.realpath(
                os.path.join(
                    os.path.dirname(__file__),
                    '..',
                    'cache'
                )
            )
        )

    def get_scan(self) -> np.array:

        """ Returns the original scan. """

        data = self._scan.get_fdata()

        # Returns Data by converting Pixel Intensities into Houndsfield Unit
        slope, intercept = self._scan.get_header().get_slope_inter()
        if slope is not None and intercept is not None:
            data = slope * data
            data += intercept

        return data

    def get_lung_mask(self) -> np.array:

        """ Returns boolean mask array which is True for all voxels being part of the lung. """

        def compute_mask(scan_data):

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
            # todo: ensure correct direction by normalizing scan data using nifti header information
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

        return self._numpy_cache.get_cached(compute_mask, self._scan_file_path, self.get_scan())

    def get_lung(self) -> Segmentation:

        """ Returns the original scan data being overlayed by the lung mask. """

        return self._get_masked_data(self.get_lung_mask())

    def get_left_lung_wing(self) -> Segmentation:

        return self._get_masked_data(self._get_lung_wing_masks()[0])

    def get_right_lung_wing(self) -> Segmentation:

        return self._get_masked_data(self._get_lung_wing_masks()[1])

    def get_vessel_mask(self) -> np.array:

        """ Returns boolean mask array which is True for all voxels being considered as blood vessels. """

        # Thresholding yields blood vessel point cloud within lung volume
        # large vessels have a HU value between -200 and +70, smallest vessels can have values of up to -570 (currently excluded due to noise)
        masked_vessel = np.copy(self.get_lung().get_data())
        masked_vessel[np.isnan(masked_vessel)] = 1000 # must not fall into thresholds below
        masked_vessel[(-500 < masked_vessel) & (masked_vessel < 70)] = 1
        masked_vessel[masked_vessel != 1] = 0

        return masked_vessel.astype(bool)

    def get_vessels(self) -> Segmentation:

        return self._get_masked_data(self.get_vessel_mask())

    def get_lung_without_vessels(self) -> Segmentation:

        return self._get_masked_data(self.get_vessel_mask() ^ self.get_lung_mask())

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

    def _get_masked_data(self, mask: np.array) -> Segmentation:

        masked_data = np.copy(self.get_scan())
        masked_data[~mask] = np.nan

        return Segmentation(self, masked_data)

    def _get_lung_wing_masks(self) -> Tuple[np.array, np.array]:
        """
        Computes and returns a tuple consisting of masks for both the left and the right lung wing.

        Returns
        -------
        masks : tuple
            The two masks as a tuple.
        """

        data = self.get_lung_mask()
        shape = data.shape

        estimated_centers = np.array([
            [
                shape[0] // 2,
                shape[1] // 2,
                shape[2] // 3,
            ],
            [
                shape[0] // 2,
                shape[1] // 2,
                shape[2] // 3 * 2,
            ]
        ])

        mask_coordinates = np.argwhere(data)

        kmeans = KMeans(n_clusters=2, init=estimated_centers, n_init=1)
        labels = kmeans.fit_predict(mask_coordinates)

        left_coordinates = mask_coordinates[labels == 0]
        left_mask = np.zeros_like(data)
        left_mask[
            left_coordinates[:, 0],
            left_coordinates[:, 1],
            left_coordinates[:, 2]
        ] = True

        right_coordinates = mask_coordinates[labels == 1]
        right_mask = np.zeros_like(data)
        right_mask[
            right_coordinates[:, 0],
            right_coordinates[:, 1],
            right_coordinates[:, 2]
        ] = True

        return left_mask, right_mask
