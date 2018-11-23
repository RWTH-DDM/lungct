
import glob
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


# Read in paths of available data
data_base_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'data'))

image_files = glob.glob(os.path.join(data_base_path, 'IMG_*'))
mask_files = [path.replace('IMG_', 'MASK_') for path in image_files]

for mask_file in mask_files:
    if not os.path.isfile(mask_file):
        raise Exception("Missing mask file " + mask_file)

print("Images found: %d" % len(image_files))

if len(image_files) == 0:
    raise Exception("No data found under " + data_base_path)


# Use first image
scan = nib.load(image_files[0]).get_fdata()
mask = nib.load(mask_files[0]).get_fdata()

print("Dimensions: " + str(scan.shape))
print("Min: " + str(np.amin(scan)))
print("Max: " + str(np.amax(scan)))


# Thresholding
# Thresholds taken from https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152505
segmentation = np.copy(scan)
segmentation[(-950 < segmentation) & (segmentation < -701)] = 1
segmentation[segmentation != 1] = 0


# Display result
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
ax1.imshow(scan[scan.shape[0] // 2])
ax1.set_title('Image')
ax2.imshow(mask[mask.shape[0] // 2])
ax2.set_title('Mask')
ax3.imshow(segmentation[segmentation.shape[0] // 2])
ax3.set_title('Segmentation')
fig.show()
