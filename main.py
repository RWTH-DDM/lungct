
import time
import lungct.data as data
import lungct.segmentation as seg
import matplotlib.pyplot as plt


# Use first image
scan = data.get_image('0002')


# Find lung
print("Segmenting out lung...")
start = time.time()
segmentation = seg.get_lunk_mask(scan)
end = time.time()
print("(Segmentation took %.2f seconds)" % (end - start))


# Display result as central slice thru the scan, the given segmentation and the computed segmentation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

scan_data = scan.get_fdata()
ax1.imshow(scan_data[scan_data.shape[0] // 2])
ax1.set_title('Image')

ax2.imshow(segmentation[segmentation.shape[0] // 2])
ax2.set_title('Segmentation')

fig.show()
