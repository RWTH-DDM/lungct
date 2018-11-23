
import glob
import os
import nibabel as nib
import matplotlib.pyplot as plt


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


# Display a central slice of the first data/mask-pair
test_image = nib.load(image_files[0]).get_data()
test_mask = nib.load(mask_files[0]).get_data()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(test_image[test_image.shape[0] // 2])
ax1.set_title('Image')
ax2.imshow(test_mask[test_mask.shape[0] // 2])
ax2.set_title('Mask')
fig.show()
