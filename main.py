
import time
import lungct.data as data
import lungct.segmentation as seg
import matplotlib.pyplot as plt
import numpy as np


# Use first image
scan = data.get_image('0002')


# Find lung
print("Segmenting out lung...")
start = time.time()
mask = seg.get_lung_mask(scan)
end = time.time()
print("(Segmentation took %.2f seconds)" % (end - start))


# Display result as central slice thru the scan, the given segmentation and the computed segmentation
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

scan_data = scan.get_fdata()
masked_data = np.copy(scan_data)
masked_data[~mask] = 0.
height = scan_data.shape[0] // 2

ax1.imshow(scan_data[height])
ax1.set_title('Original image')

ax2.imshow(mask[height])
ax2.set_title('Mask')

ax3.imshow(masked_data[height])
ax3.set_title('Masked image')

fig.show()
