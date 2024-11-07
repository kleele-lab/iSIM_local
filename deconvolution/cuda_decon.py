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


# def main():
#     from Analysis.tools import get_files
#     folder = '/nfs/nas22/fs2202/biol_bc_kleele_2/Joshua/240119_RPE1_Torin_Mdivi_MFI8_iSIM/18h'
#     files, _ = get_files(folder)
#
#     algo = fd_restoration.RichardsonLucyDeconvolver(2).initialize()
#     kernel = make_kernel(io.imread(files['network'][0]), sigma=3.9/2.355)
#     for struct_file in files['network']:
#         struct_img = io.imread(struct_file)
#         _, axs = plt.subplots(1, 2)
#         # axs[0].imshow(struct_img)
#         t0 = time.perf_counter()
#         struct_img = prepare_decon(struct_img)
#         img = richardson_lucy(struct_img, algo=algo, kernel=kernel).astype(np.uint16)
#         print(time.perf_counter()-t0)
#         axs[0].imshow(struct_img.astype(np.uint16))
#         axs[1].imshow(img)
#         # plt.show()


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
    print("=====", size)
    size = [min([17, x]) for x in size]
    print("=====", size)

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

                print("---", data_c.shape)

                for ij in range(0, data_c.shape[0]):
                    params.kernel = make_kernel(image=data_c[ij, :, :], sigma=params.sigma, z_step=params.z_step)
                    maxval_slice = np.max(data_c[ij, :, :])
                    result = restoration.richardson_lucy(data_c[ij, :, :] / maxval_slice,
                                                         psf=params.kernel['kernel'],
                                                         num_iter=30)  #
                    # plt.imshow(result, vmin=result.min(), vmax=result.max())
                    decon[timepoint, ij, channel, :, :] = result * maxval_slice

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

    # UUID = uuid.uuid1()

    filename = str(file_dir.split('.ome.tif')[0]) + '_metadata.txt'
    f_metadata = open(filename, 'r')
    # metadata_text = f_metadata.read()
    metadata_json = json.load(f_metadata)
    # Naive attempt to save as tiff
    f_metadata.close()
    io.imsave(os.path.join(os.path.dirname(file_dir), out_file_tiff), decon, metadata=imagej_metadata)


    # Get metadata to transfer

#    with tifffile.TiffReader(file_dir) as reader:
#        mdInfo = xmltodict.parse(reader.ome_metadata)
#        #mdInfo['OME']['Image']["Pixels"]["@DimensionOrder"] = "XYZCT"
#        mdInfo['OME']['Image']['@Name'] = os.path.basename(out_file).split('.')[0]     
#        for frame_tiffdata in mdInfo['OME']['Image']['Pixels']['TiffData']:
#            frame_tiffdata['UUID']['@FileName'] = os.path.basename(out_file)
#            frame_tiffdata['UUID']['#text'] =  'urn:uuid:' + str(UUID)

#    with tifffile.TiffWriter(os.path.join(os.path.dirname(file_dir), out_file_tiff), byteorder='<', ome=True) as tif: 
#        tif.ome_metadata = old_metadata
#        tif.write(decon, 
#                photometric='minisblack', 
#                rowsperstrip=1532, 
#                bitspersample=16, 
#                compression='None', 
#                resolution=(219780, 219780, 'CENTIMETER'),
#                metadata=imagej_metadata #{'ImageJ':'1.51s','axes':'TZCYX', 'mode':'composite', 'unit': 'um','Ranges': (190.0, 18780.0, 188.0, 1387.0)}, #,'LUTs': imagej_metadata['LUTs'], 'IJMetadataByteCounts': (28, 2116, 32, 768, 768) }, #'spacing': 0.1499999999999999, 'unit': 'um','Ranges': (190.0, 18780.0, 188.0, 1387.0), 'IJMetadataByteCounts': (28, 2116, 32, 768, 768) },
#                extratags=[(50838,'int',5,(28, 2116, 32, 768, 768),True),
#                    (5089,'str',None,imagej_metadata,True),
#                    (279,'int', 2,(6556960,),True),
#                    (286,'float',1, 12342.2, True),
#                    (287,'float',1, -6171.9, True),
#                    (286,'float',1, 12342.2, True)
#                    ]
#                )

### Commented out due to working io.imsave as ome-TIFF above -> saving data storage
# hf = h.File(os.path.join(os.path.dirname(file_dir), out_file[0] + "_decon.h5"), 'w')
# hf.create_dataset('data',data=decon)
# hf.close()

# Write metadata to the prepared file
#    my_mdInfo = xmltodict.unparse(mdInfo).encode(encoding='UTF-8', errors='strict')
#    tifffile.tiffcomment(os.path.join(os.path.dirname(file_dir), out_file), comment=imagej_metadata) # my_mdInfo)

# with tifffile.TiffWriter(os.path.join(os.path.dirname(file_dir), out_file), bigtiff=True) as tif:
#     with tifffile.TiffReader(file_dir) as reader:
#         mdInfo = xmltodict.parse(reader.ome_metadata)
#         for frame_tiffdata in mdInfo['OME']['Image']['Pixels']['TiffData']:
#             frame_tiffdata['UUID']['@FileName'] = os.path.basename(out_file)
#         mdInfo = xmltodict.unparse(mdInfo).encode(encoding='UTF-8', errors='strict')
#     tif.write(decon, description=mdInfo)


# def decon_one_frame(file_dir, params=None):
#     image = tifffile.imread(file_dir)
#
#     if params is None:
#         params = {'background': 'median'}
#
#     # image = cv2.GaussianBlur(image, (0, 0), 1)
#     print("Image shape for decon", image.shape)
#     kernel_shape = image.shape
#     ndim = 2
#     params = CudaParams(background=params['background'], shape=kernel_shape, ndim=ndim)
#     decon = richardson_lucy(image, params=params)
#
#     out_file = os.path.basename(file_dir).rsplit('.', 2)
#     out_file = out_file[0] + ".".join(["_decon", *out_file[1:]])
#
#     tifffile.imwrite(os.path.join(os.path.dirname(file_dir), out_file), decon)
