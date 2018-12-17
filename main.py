
import lungct.data as data
import matplotlib.pyplot as plt
from LungCT import LungCT


lungct = LungCT(data.get_image_path('0002'))

print("Volume: %f l" % lungct.get_volume())
print("PD 5%%: %f" % lungct.get_percentile_density(5))
print("PD 95%%: %f" % lungct.get_percentile_density(95))
print("Average: %f" % lungct.get_average_density())
print("Median: %f" % lungct.get_median_density())


# Display result as central slice thru the scan, the given segmentation and the computed segmentation
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

scan_data = lungct.get_scan()
mask = lungct.get_mask()
lung = lungct.get_lung()
height = scan_data.shape[0] // 2

ax1.imshow(scan_data[height])
ax1.set_title('Original image')

ax2.imshow(mask[height])
ax2.set_title('Mask')

ax3.imshow(lung[height])
ax3.set_title('Masked image')

fig.show()
