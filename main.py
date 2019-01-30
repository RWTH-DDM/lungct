
import lungct.data as data
import matplotlib.pyplot as plt
from lungct.LungCT import LungCT


def show_slice(data, points=None):

    fig, ax = plt.subplots()
    ax.imshow(data[data.shape[0] // 2])

    if points:
        for point in points:
            ax.plot(point[0], point[1], 'ro')

    fig.show()

def show_three_slices(data, points=None):

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    shape = data.shape

    ax1.imshow(data[:, :, shape[2] // 2])
    ax1.set_title('X-plane')

    ax2.imshow(data[:, shape[1] // 2, :])
    ax2.set_title('Y-plane')

    ax3.imshow(data[shape[0] // 2, :, :])
    ax3.set_title('Z-plane')

    """if points:
        for point in points:
            ax1.plot(point[])"""

    fig.show()

def show_distribution(data):

    fig, axis = plt.subplots(fig=(6, 6))
    axis.hist(data)
    fig.show()


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


show_three_slices(lungct.get_scan())
show_three_slices(lungct.get_lung().get_data())
show_three_slices(lungct.get_vessel().get_data())