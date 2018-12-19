
import lungct.data as data
import matplotlib.pyplot as plt
from LungCT import LungCT


lungct = LungCT(data.get_image_path('0002'))

print("Volume: %f l" % lungct.get_volume())
print("PD 5%%: %f" % lungct.get_percentile_density(5))
print("PD 95%%: %f" % lungct.get_percentile_density(95))
print("Average: %f" % lungct.get_average_density())
print("Median: %f" % lungct.get_median_density())


# Display result as central slice through the scan, the given segmentation and the computed segmentation
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

scan_data = lungct.get_scan()
lung = lungct.get_lung()
vessel_mask = lungct.get_vessel_mask()
lung_without_vessels = lungct.get_lung_without_vessels()

height = scan_data.shape[0] // 2

ax1.imshow(scan_data[height])
ax1.set_title('Original image')

ax2.imshow(lung[height])
ax2.set_title('Masked image')

ax3.imshow(vessel_mask[height])
ax3.set_title('Masked Vessels')

ax4.imshow(lung_without_vessels[height])
ax4.set_title('Lung Without Vessels')

fig.show()
