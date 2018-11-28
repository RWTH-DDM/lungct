
import lungct.data as data
import lungct.segmentation as seg
import matplotlib.pyplot as plt


# Use first image
scan = data.get_image('0002')
mask = data.get_mask('0002')


# Find lung
segmentation = seg.segment_lung(scan)


# Display result as central slice thru the scan, the given segmentation and the computed segmentation
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

scan_data = scan.get_fdata()
ax1.imshow(scan_data[scan_data.shape[0] // 2])
ax1.set_title('Image')

mask_data = mask.get_fdata()
ax2.imshow(mask_data[mask_data.shape[0] // 2])
ax2.set_title('Mask')

ax3.imshow(segmentation[segmentation.shape[0] // 2])
ax3.set_title('Segmentation')

fig.show()
