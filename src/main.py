
import matplotlib.pyplot as plt

from lungct.data import *
from lungct.segmentation import *


# Use first image
scan = get_image('0002')
mask = get_mask('0002')

print("Dimensions: " + str(scan.shape))


segmentation = segment_lung(scan)


# Display result
scan_data = scan.get_fdata()
mask_data = mask.get_fdata()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
ax1.imshow(scan_data[scan_data.shape[0] // 2])
ax1.set_title('Image')
ax2.imshow(mask_data[mask_data.shape[0] // 2])
ax2.set_title('Mask')
ax3.imshow(segmentation[segmentation.shape[0] // 2])
ax3.set_title('Segmentation')
fig.show()
