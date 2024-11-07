from typing import Tuple, Callable
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
from skimage import io
from skimage import restoration
from scipy import ndimage
import numpy as np
from prepare import prepare_decon, get_filter_zone
from dataclasses import dataclass
import matplotlib.pyplot as plt

import tifffile
import xmltodict
# from dicttoxml import dicttoxml
from tqdm import tqdm
from typing import Union
import json
# from PIL import Image

import uuid
# import time
import os

import pdb

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = 'platform'

SIZE_LIMIT = 100_000_000
OVERLAP = 6

@dataclass
class Params():
    """ Class for storing parameters to be used in deconvolution"""
    sigma: float = 3.9 / 2.335
    z_step: float = 0.2
    background: Union[int, str] = 'median'
    after_gaussian: float = 2
    kernel = None

    def to_dict(self):
        class_dict = {'sigma': self.sigma,
                      'z_step': self.z_step,
                      'background': self.background,
                      'after_gaussian': self.after_gaussian}
        return class_dict


def make_kernel(image: np.ndarray, sigma=1.67, z_step=0.2):
    """Make a gaussian kernel that fits the psf of the microscope"""
    if image.ndim == 3:
        z_sigma = 0.48 / z_step
        sigma = [z_sigma, sigma, sigma]
        print("3D images, sigma: ", sigma)

    size = image.shape
    size = [min([17, x]) for x in size]

    # If even size dimensions crop to have a center pixel
    kernel = {'kernel': np.zeros(size, dtype=float),
              'sigma': sigma}
    kernel['kernel'][tuple(np.array(kernel['kernel'].shape) // 2)] = 1
    kernel['kernel'] = ndimage.gaussian_filter(kernel['kernel'], sigma=sigma)

    kernel['kernel'][kernel['kernel'] < 1e-6 * np.max(kernel['kernel'])] = 0
    kernel['kernel'] = np.divide(kernel['kernel'], np.sum(kernel['kernel'])).astype(np.float32)
    return kernel


def decon_ome_stack(file_dir, params=None):
    data = None
    params = Params()
    with tifffile.TiffFile(file_dir) as tif:# , is_mmstack=False, is_ome=True
        assert tif.is_imagej
        imagej_metadata = tif.imagej_metadata


        my_dict = xmltodict.parse(tif.ome_metadata, force_list={'Plane'})
        size_t = int(my_dict['OME']['Image']["Pixels"]["@SizeT"])
        size_z = int(my_dict['OME']['Image']["Pixels"]["@SizeZ"])
        size_c = int(my_dict['OME']['Image']["Pixels"]["@SizeC"])
        try:
            z_step = float(my_dict['OME']['Image']["Pixels"]['@PhysicalSizeZ'])
        except KeyError:
            print("Could not get z step size. Will put default 0.2")
            z_step = 0.2
        # 'XYCZT' or 'XYZCT' ?
        dim_order = my_dict['OME']['Image']["Pixels"]["@DimensionOrder"]

        data = tif.asarray()

    # if data is None:
    #     print("ATTENION: NORMAL READING OF TIFF FAILED! RESORT TO BASIC! ASSUME 1 TIME POINT & 1 CHANNEL!")
    #     data = io.imread(file_dir, plugin='pil')
    #
    #     dim_order = 'XYCZT'
    #
    #     size_t = 1
    #     size_z = data.shape[0]
    #     size_c = 1
    #
    #     z_step = 0.2
    print("\n Sizes : ", size_t, size_z, size_c)
    print("Dim_order: ", dim_order)

    ndim = 2 if size_z == 1 else 3
    # Make standardized array with all dimensions
    # time [0], z[1], c[2] if not there, just missing depending on order
    # If it was recorded differently, swap dimensions

    if size_t == 1:
        data = np.expand_dims(data, 0)

    if size_z > 1 and size_c > 1 and dim_order == 'XYZCT':
        data = np.moveaxis(data, 1, 2)

    if size_z == 1:
        data = np.expand_dims(data, 1)
    if size_c == 1:
        data = np.expand_dims(data, 2)

    if size_t != data.shape[0]:
        size_t = data.shape[0]
    print("SHAPE", data.shape)

    # Make data odd shaped
    original_size_data = data.shape
    pad = tuple(np.zeros((5, 2), int))
    crop = tuple(np.zeros((5, 2), int))
    if data.shape[1] % 2 == 0:
        pad_here = tuple(np.zeros((5, 2), int))
        pad_here[1][1] = 1
        data = np.pad(data, pad)
    for dim in [3, 4]:
        if data.shape[dim] % 2 == 0:
            pad[dim][1] = 1
            crop[dim][1] = data.shape[dim] - 1
        else:
            crop[dim][1] = data.shape[dim]
    data = data[:, :, :, :crop[3][1], :crop[4][1]]

    kernel_shape = data.shape[-2:] if ndim == 2 else [np.min([17, size_z]), *data.shape[-2:]]
    # Decon
    if params is None:
        background = 100
    else:
        background = params.background

    my_slices = None
    decon = np.empty_like(data)
    for timepoint in tqdm(range(size_t)):
        data_t = data[timepoint, :, :, :, :]
        for channel in range(size_c):
            data_c = data_t[:, channel, :, :]
            if size_z == 1:
                data_c = data_c[0, :, :]

            if my_slices is None:
                # if ndim == 3:

                # padding = (data_c.shape[0] - params.kernel['kernel'].shape[0]) // 2
                # params.kernel['kernel'] = np.pad(params.kernel['kernel'], ((padding, padding), (0, 0), (0, 0)))
                params.kernel = make_kernel(image=data_c[0, :, :], sigma=params.sigma, z_step=params.z_step)
                for z_i in tqdm( range(0, data_c.shape[0]) ):
                #for ij in range(0, data_c.shape[0]):
                    maxval_slice = np.max(data_c[z_i, :, :])
                    result = restoration.richardson_lucy(data_c[z_i, :, :] / maxval_slice,
                                                         psf=params.kernel['kernel'],
                                                         num_iter=30)  #
                    # plt.imshow(result, vmin=result.min(), vmax=result.max())
                    decon[timepoint, z_i, channel, :, :] = result * maxval_slice

                    # pdb.set_trace()

    # Crop the data back if we padded it
    if original_size_data != decon.shape:
        decon = decon[:, :original_size_data[1], :, :, :]
    if original_size_data != decon.shape:
        decon = np.pad(decon, pad)
    print("DECON SHAPE ", decon.shape)





    # Swap axes back
    # print("DECON", decon.shape)
    # if size_z > 1 and size_c > 1 and dim_order == 'XYZCT':
    #     decon = np.moveaxis(decon, 1, 2)
    # decon = np.squeeze(decon)
    # print("DECON READY", decon.shape)
    # Output

    # Adjust metadata
    #    try:
    #        imagej_metadata['min'] = np.min(decon)
    #        imagej_metadata['max'] = np.max(decon)
    #        imagej_metadata['Ranges'] = (np.min(decon), np.max(decon))
    #        for idx, lut in enumerate(imagej_metadata['LUTs']):
    #            imagej_metadata['LUTs'][idx] = imagej_metadata['LUTs'][idx].tolist()
    #    except TypeError:
    #        print("Could not set imagej_metadata")
    # info = json.loads(imagej_metadata['Info'])
    # Construct the correct one here
    # info['AxisOrder'] = ['position', 'time', 'z', 'channel']
    # imagej_metadata['Info'] = json.dumps(info)

    out_file = os.path.basename(file_dir).rsplit('.', 2)
    out_file_tiff = out_file[0] + ".".join(["_decon", *out_file[1:]])

    # filename = str(file_dir.split('.ome.tif')[0]) + '_metadata.txt'
    # f_metadata = open(filename, 'r')
    # # metadata_text = f_metadata.read()
    # metadata_json = json.load(f_metadata)
    #
    # f_metadata.close()

    with tifffile.TiffWriter(os.path.join(os.path.dirname(file_dir), out_file_tiff), imagej=True) as dst:
        for decon_one in decon:
            frame = decon_one
            dst.write(
                frame,
                contiguous=True,
                metadata=imagej_metadata,
            )

    #io.imsave(os.path.join(os.path.dirname(file_dir), out_file_tiff), decon, metadata=imagej_metadata)
