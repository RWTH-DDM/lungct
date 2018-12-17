
import glob
import os
import re


_data_base_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'data'))


def get_image_ids():

    image_files = glob.glob(os.path.join(_data_base_path, 'IMG_*'))

    print("Images found: %d" % len(image_files))

    if len(image_files) == 0:
        raise Exception("No data found under " + _data_base_path)

    return [re.search(r'IMG_(\d+)\.nii\.gz$', file_name).group(1) for file_name in image_files]


def get_image_path(id):

    return os.path.join(_data_base_path, 'IMG_' + id + '.nii.gz')

