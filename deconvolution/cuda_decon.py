import posixpath

from skimage import restoration
from scipy import ndimage
import numpy as np
from dataclasses import dataclass
import tifffile
import xmltodict
from tqdm import tqdm
from typing import Union
import os
import javabridge as jb
import bioformats as bf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = 'platform'
def _init_logger():
    """This is so that Javabridge doesn't spill out a lot of DEBUG messages
    during runtime.
    From CellProfiler/python-bioformats.
    """
    rootLoggerName = jb.get_static_field("org/slf4j/Logger",
                                         "ROOT_LOGGER_NAME",
                                         "Ljava/lang/String;")

    rootLogger = jb.static_call("org/slf4j/LoggerFactory",
                                "getLogger",
                                "(Ljava/lang/String;)Lorg/slf4j/Logger;",
                                rootLoggerName)

    logLevel = jb.get_static_field("ch/qos/logback/classic/Level",
                                   "WARN",
                                   "Lch/qos/logback/classic/Level;")

    jb.call(rootLogger,
            "setLevel",
            "(Lch/qos/logback/classic/Level;)V",
            logLevel)


@dataclass
class Params():
    """ Class for storing parameters to be used in deconvolution"""
    sigma: float = 3.9 / 2.335
    z_step: float = 0.2
    background: Union[int, str] = 'median'
    kernel = None


class Image():
    """ Generic class to store the data for the deconvolution 
    Image can be read from tif or vsi"""

    def __init__(self):
        data: np.ndarray = np.empty(1)
        metadata: dict = {}
        z_step: float = 0.2
        size_t: int = 0
        size_c: int = 0
        size_z: int = 0
        out_file_end: str = ''

    def read_tiff(self, file_dir: posixpath) -> None:
        tif = tifffile.TiffFile(file_dir)
        Image.metadata = tif.imagej_metadata

        my_dict = xmltodict.parse(tif.ome_metadata, force_list={'Plane'})
        size_t = int(my_dict['OME']['Image']["Pixels"]["@SizeT"])
        size_z = int(my_dict['OME']['Image']["Pixels"]["@SizeZ"])
        size_c = int(my_dict['OME']['Image']["Pixels"]["@SizeC"])

        try:
            Image.z_step = float(my_dict['OME']['Image']["Pixels"]['@PhysicalSizeZ'])
        except KeyError:
            print("Could not get z step size. Will put default 0.2")
            Image.z_step = 0.2

        # 'XYCZT' or 'XYZCT' ?
        dim_order = my_dict['OME']['Image']["Pixels"]["@DimensionOrder"]
        Image.data = tif.asarray()

        # This is legacy code to handle tif dimension issues

        # if data is None:
        #     print("ATTENTION: NORMAL READING OF TIFF FAILED! RESORT TO BASIC! ASSUME 1 TIME POINT & 1 CHANNEL!")
        #     data = io.imread(file_dir, plugin='pil')
        #
        #     dim_order = 'XYCZT'
        #
        #     size_t = 1
        #     size_z = data.shape[0]
        #     size_c = 1
        #
        #     z_step = 0.2

        ndim = 2 if size_z == 1 else 3
        # Make standardized array with all dimensions
        # time [0], z[1], c[2] if not there, just missing depending on order
        # If it was recorded differently, swap dimensions

        if size_t == 1:
            Image.data = np.expand_dims(Image.data, 0)

        if size_z > 1 and size_c > 1 and dim_order == 'XYZCT':
            Image.data = np.moveaxis(Image.data, 1, 2)

        if size_z == 1 and len(Image.data.shape) < 5:
            Image.data = np.expand_dims(Image.data, 1)
        if size_c == 1 and len(Image.data.shape) < 5:
            Image.data = np.expand_dims(Image.data, 2)

        if size_t != Image.data.shape[0]:
            size_t = Image.data.shape[0]

        Image.size_z = size_z
        Image.size_t = size_t
        Image.size_c = size_c

        out_file = os.path.basename(file_dir).rsplit('.', 2)
        Image.out_file_end = out_file[0] + ".".join(["_decon", *out_file[1:]])

        print("\nReading tif:")
        print("Sizes, t, z, and c : ", size_t, size_z, size_c)
        print("Dim_order in the original file : ", dim_order)
        print("New shape of data going into decon", Image.data.shape, "\n")

    def read_vsi(self, file_dir: posixpath) -> None:
        jb.start_vm(class_path=bf.JARS)
        logger = _init_logger()

        vsi = bf.ImageReader(file_dir)
        txt_metadata = bf.get_omexml_metadata(file_dir)
        txt_metadata = txt_metadata.encode('ascii', errors="ignore")

        my_dict = xmltodict.parse(txt_metadata, force_list={'Plane'})
        Image.metadata = my_dict

        size_t = int(my_dict['OME']['Image'][0]['Pixels']['@SizeT'])
        size_z = int(my_dict['OME']['Image'][0]['Pixels']['@SizeZ'])
        size_c = int(my_dict['OME']['Image'][0]['Pixels']['@SizeC'])
        size_x = int(my_dict['OME']['Image'][0]['Pixels']['@SizeX'])
        size_y = int(my_dict['OME']['Image'][0]['Pixels']['@SizeY'])

        dim_order = my_dict['OME']['Image'][0]['Pixels']['@DimensionOrder']

        try :
            Image.z_step = float(my_dict['OME']['Image'][0]["Pixels"]['@PhysicalSizeZ'])
        except KeyError:
            print("Could not get z step size. Will put default 0.2")
            Image.z_step = 0.2

        if dim_order == 'XYCZT':
            # data = np.zeros((size_t, size_z, size_c, size_y, size_x))
            data = np.zeros((size_x, size_y, size_c, size_z, size_t))
            for i_z in range(0, size_z):
                for i_t in range(0, size_t):
                    data[:, :, :, i_z, i_t] = vsi.read(series='0', z=i_z, t=i_t)
        else:
            raise "Dim order is NOT XYZCT"

        Image.data = np.moveaxis(data, [0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
        Image.size_z = size_z
        Image.size_t = size_t
        Image.size_c = size_c

        out_file = os.path.basename(file_dir).rsplit('.', 2)
        Image.out_file_end = out_file[0] + ".".join(["_decon.tif "])
        jb.kill_vm()
        print("\nReading vsi:")
        print("Sizes, t, z, and c : ", size_t, size_z, size_c)
        print("Dim_order in the original file : ", dim_order)
        print("New shape of data going into decon", Image.data.shape, "\n")

def make_kernel(image: np.ndarray, sigma=1.67, z_step=0.2):
    """
    Make a gaussian kernel that fits the psf of the microscope.

    The psf is designed to have the same number of dimensions then
    the image. The maximum shape of the psf is 100 in either dimension,
    this limit has been arbitrarily chosen. From tests, higher psf
    dimensions lead to high processing times and RAM consumptions, without
    visually improving on the results.
    """

    if image.ndim == 3:
        z_sigma = 0.48 / z_step
        sigma = [z_sigma, sigma, sigma]

    size = image.shape
    size = [min([100, x]) for x in size]

    # If even size dimensions crop to have a center pixel
    kernel = {'kernel_array': np.zeros(size, dtype=float),
              'sigma': sigma}
    kernel['kernel_array'][tuple(np.array(kernel['kernel_array'].shape) // 2)] = 1
    kernel['kernel_array'] = ndimage.gaussian_filter(kernel['kernel_array'], sigma=sigma)

    kernel['kernel_array'][kernel['kernel_array'] < 1e-6 * np.max(kernel['kernel_array'])] = 0
    kernel['kernel_array'] = np.divide(kernel['kernel_array'], np.sum(kernel['kernel_array'])).astype(np.float32)
    return kernel


def get_data_c(data_t, size_c, size_z):
    """
    Ensures a progressive supply of images to the deconvolution routine.

    Will provide z-stack of images, per channel.
    """

    for channel in range(size_c):
        data_c = data_t[:, channel, :, :]
        # if size_z == 1:
        #     data_c = data_c[0, :, :]

        yield channel, data_c


def decon_ome_stack(file_dir, background):
    """
    Main deconvolution routine that reads in images, processes,
    deconvolves, and saves the data.

    """

    params = Params
    params.background = background

    Img = Image()
    if file_dir.split('.')[-1] == "tif":
        Img.read_tiff(file_dir)

    elif file_dir.split('.')[-1] == "vsi":
        Img.read_vsi(file_dir)

    # Make data odd shaped
    original_size_data = Img.data.shape
    pad = tuple(np.zeros((5, 2), int))
    crop = tuple(np.zeros((5, 2), int))

    if Img.data.shape[1] % 2 == 0:
        pad_here = tuple(np.zeros((5, 2), int))
        pad_here[1][1] = 1
        Img.data = np.pad(Img.data, pad)
    for dim in [3, 4]:
        if Img.data.shape[dim] % 2 == 0:
            pad[dim][1] = 1
            crop[dim][1] = Img.data.shape[dim] - 1
        else:
            crop[dim][1] = Img.data.shape[dim]
    Img.data = Img.data[:, :, :, :crop[3][1], :crop[4][1]]

    # start of the deconvolution loop
    decon = np.empty_like(Img.data)
    for timepoint in tqdm(range(Img.size_t)):
        data_t = Img.data[timepoint, :, :, :, :]
        data_c_iterable = get_data_c(data_t, Img.size_c, Img.size_z)

        for channel, data_c in data_c_iterable:
            params.kernel = make_kernel(image=data_c, sigma=params.sigma, z_step=params.z_step)
            maxval_uint16 = 65535  # max value for 16-bit images;  alternative: np.max(data_c)

            # let't pad the array in x and y directions
            pad_value = 50
            origin_shape = list(data_c.shape)
            target_shape = np.copy(origin_shape)
            target_shape[1] += 2 * pad_value
            target_shape[2] += 2 * pad_value

            for_decon = np.zeros(target_shape) + np.median(data_c) / maxval_uint16

            if np.max(data_c) > 1:
                for_decon[:,
                pad_value:origin_shape[1] + pad_value,
                pad_value:origin_shape[2] + pad_value] = data_c / maxval_uint16
            else:
                for_decon[:,
                pad_value:origin_shape[1] + pad_value,
                pad_value:origin_shape[2] + pad_value] = data_c

            result = restoration.richardson_lucy(for_decon,
                                                 psf=params.kernel['kernel_array'],
                                                 num_iter=10)

            result = result * maxval_uint16

            decon[timepoint, :, channel, :, :] = result.astype(np.uint16)[:,
                                                 pad_value:origin_shape[1] + pad_value,
                                                 pad_value:origin_shape[2] + pad_value]

    # Here the main tif file is closed : post-processing and saving the deconvolved images next

    # Crop the data back if we padded it
    if original_size_data != decon.shape:
        decon = decon[:, :original_size_data[1], :, :, :].astype(np.uint16)
    if original_size_data != decon.shape:
        decon = np.pad(decon, pad).astype(np.uint16)

    print("Shape of deconvoluted data  : ", decon.shape)

    with tifffile.TiffWriter(os.path.join(os.path.dirname(file_dir), Img.out_file_end), imagej=True) as dst:
        for decon_one in decon:
            frame = decon_one
            dst.write(
                frame,
                contiguous=True,
                metadata=Img.metadata,
                dtype=np.uint16
            )
