
import lungct.data as data
import matplotlib.pyplot as plt
from lungct.LungCT import LungCT


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
print("Density gradient: %s" % (lung_with_vessels.get_density_gradient(),))

print("\nLung without vessels:")
print("Volume: %f l" % lung_without_vessels.get_volume())
print("PD 5%%: %f" % lung_without_vessels.get_percentile_density(5))
print("PD 95%%: %f" % lung_without_vessels.get_percentile_density(95))
print("Average: %f" % lung_without_vessels.get_average_density())
print("Median: %f" % lung_without_vessels.get_median_density())
print("Density gradient: %s" % (lung_without_vessels.get_density_gradient(),))


print("\nComputing distances...")
to_vessel_distances = lung_without_vessels.get_distances_to_nearest(vessels)


# Display result as central slice through the scan, the given segmentation and the computed segmentation
print("\nDisplaying slices...")
fig, [[ax11, ax12, ax13], [ax21, ax22, ax23], [ax31, ax32, ax33]] = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))

scan_data = lungct.get_scan()
height = scan_data.shape[0] // 2

ax11.imshow(scan_data[height])
ax11.set_title('Original image')

ax12.imshow(lungct.get_mask()[height])
ax12.set_title('Lung mask')

ax13.imshow(lungct.get_lung().get_data()[height])
ax13.set_title('Masked image')

ax21.axis('off')

ax22.imshow(lungct.get_vessel_mask()[height])
ax22.set_title('Masked Vessels')

ax23.imshow(lungct.get_lung_without_vessels().get_data()[height])
ax23.set_title('Lung Without Vessels')

ax31.hist(to_vessel_distances)

ax32.axis('off')
ax33.axis('off')

fig.show()
