
import lungct.data as data
import matplotlib.pyplot as plt
from LungCT import LungCT


lungct = LungCT(data.get_image_path('0002'))
lung_with_vessels = lungct.get_lung()
lung_without_vessels = lungct.get_lung_without_vessels()
vessels = lungct.get_vessel()

# export segmentations
print("Exporting segmentations...")
lung_with_vessels.nifti_export('with_vessels.nii')
lung_without_vessels.nifti_export('without_vessels.nii')
vessels.nifti_export('vessels.nii')

print("\nLung with vessels:")
print("Volume: %f l" % lung_with_vessels.get_volume())
print("PD 5%%: %f" % lung_with_vessels.get_percentile_density(5))
print("PD 95%%: %f" % lung_with_vessels.get_percentile_density(95))
print("Average: %f" % lung_with_vessels.get_average_density())
print("Median: %f" % lung_with_vessels.get_median_density())

print("\nLung without vessels:")
print("Volume: %f l" % lung_without_vessels.get_volume())
print("PD 5%%: %f" % lung_without_vessels.get_percentile_density(5))
print("PD 95%%: %f" % lung_without_vessels.get_percentile_density(95))
print("Average: %f" % lung_without_vessels.get_average_density())
print("Median: %f" % lung_without_vessels.get_median_density())


# Display result as central slice through the scan, the given segmentation and the computed segmentation
print("\nDisplaying slices...")
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

scan_data = lungct.get_scan()
height = scan_data.shape[0] // 2

ax1.imshow(scan_data[height])
ax1.set_title('Original image')

ax2.imshow(lungct.get_lung().get_data()[height])
ax2.set_title('Masked image')

ax3.imshow(lungct.get_vessel_mask()[height])
ax3.set_title('Masked Vessels')

ax4.imshow(lungct.get_lung_without_vessels().get_data()[height])
ax4.set_title('Lung Without Vessels')

fig.show()
